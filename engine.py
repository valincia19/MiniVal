"""
MiniVal High-Efficiency Training Engine
========================================
Mesin pelatihan terpadu dan profesional untuk Pretraining, SFT, LoRA, dan DPO.

Fitur:
  - Linear Warmup + Cosine Decay Learning Rate Scheduler
  - Weight Decay separation (tidak decay pada RMSNorm & bias)
  - Gradient Accumulation & Gradient Clipping
  - Mixed Precision (bfloat16 / float16) otomatis
  - Tracking metrik real-time: Loss, PPL (Perplexity), Token/sec, ETA
  - Safe-save pada interupsi (Ctrl+C)
"""
from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from config import TrainConfig
from model import apply_lora, save_lora


# Helper Loss & Optimizer

def dpo_loss_fn(
    ref_log_probs: torch.Tensor,
    policy_log_probs: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Direct Preference Optimization (DPO) loss.
    L_DPO = -E[log sigma(beta * (log(pi(y_w|x)/ref(y_w|x)) - log(pi(y_l|x)/ref(y_l|x))))]
    """
    ref_lp = (ref_log_probs * mask).sum(dim=1)
    pol_lp = (policy_log_probs * mask).sum(dim=1)
    bsz = ref_lp.shape[0]

    chosen_ref, reject_ref = ref_lp[: bsz // 2], ref_lp[bsz // 2 :]
    chosen_pol, reject_pol = pol_lp[: bsz // 2], pol_lp[bsz // 2 :]

    pi_diff = chosen_pol - reject_pol
    ref_diff = chosen_ref - reject_ref
    return -F.logsigmoid(beta * (pi_diff - ref_diff)).mean()


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    fused: bool = False,
) -> optim.AdamW:
    """
    Bangun AdamW dengan pemisahan parameter weight decay.
    Standar LLM: parameter 1D (RMSNorm weights, bias) TIDAK di-decay.
    fused=True (CUDA) -> 1 kernel per step, jauh lebih cepat.
    """
    decay_params = []
    nodecay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases dan normalisasi 1D tidak di-decay
        if param.dim() < 2 or "norm" in name.lower():
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return optim.AdamW(param_groups, lr=lr, betas=betas, eps=1e-8, fused=fused)


# MiniValTrainer

class MiniValTrainer:
    """
    Trainer terpadu untuk semua pipeline MiniVal.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        config: TrainConfig,
        ref_model: Optional[torch.nn.Module] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.cfg = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Pengaman CPU: jangan pakai 100% thread agar Windows tidak freeze
        if "cpu" in self.device:
            total_cores = os.cpu_count() or 4
            safe_threads = max(1, min(4, total_cores // 2))
            torch.set_num_threads(safe_threads)
            print(f" Mode CPU Terdeteksi: Dibatasi ke {safe_threads} threads (dari {total_cores} cores) agar PC tetap adem & responsif.")

        self.model.to(self.device)
        if self.ref_model:
            self.ref_model.to(self.device).eval()

        is_cuda = "cuda" in self.device

        # Pad token resolved sekali - `or` salah karena pad id bisa 0 (0 falsy -> fallback)
        self.pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if self.pad_id is None:
            self.pad_id = getattr(self.tokenizer, "eos_token_id", None) or 0

        os.makedirs(self.cfg.save_dir, exist_ok=True)

        # LoRA setup jika stage == 'lora'
        if self.cfg.stage == "lora":
            for p in self.model.parameters():
                p.requires_grad = False
            apply_lora(self.model, rank=self.cfg.lora_rank)

        # Optimizer dengan decay separation (+fused di CUDA: 1 kernel per step)
        self.optimizer = build_optimizer(
            self.model,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            fused=is_cuda,
        )

        # Precision context - tentukan dtype dulu
        if is_cuda:
            use_bf16 = torch.cuda.is_bf16_supported()
            dtype = torch.bfloat16 if use_bf16 else torch.float16
            self.autocast_ctx = torch.amp.autocast("cuda", dtype=dtype)
        else:
            self.autocast_ctx = nullcontext()
            dtype = None

        # Mixed Precision Scaler - bf16 TIDAK butuh scaler (loss scale redundan + overhead)
        scaler_enabled = is_cuda and dtype == torch.float16
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    def _get_lr(self, current_step: int, total_steps: int) -> float:
        """
        Linear Warmup + Cosine Decay.
        - Warmup: 0 -> lr selama warmup_steps
        - Cosine: lr -> 0.1 * lr hingga total_steps
        """
        warmup_steps = max(int(total_steps * self.cfg.warmup_ratio), 1)
        if current_step < warmup_steps:
            return self.cfg.lr * (current_step / warmup_steps)
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return self.cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

    def fit(self, dataset):
        """Mulai pelatihan untuk Pretrain, SFT, atau LoRA."""
        is_cuda = "cuda" in self.device
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=is_cuda,
            # Worker proses paralel: tokenisasi/collate tidak memblok GPU
            num_workers=2 if is_cuda else 0,
            persistent_workers=is_cuda,
            prefetch_factor=2 if is_cuda else None,
            # Collate dynamic WAJIB selalu aktif (dataset mengembalikan panjang variabel)
            collate_fn=(dataset.get_collate() if hasattr(dataset, "get_collate") else None),
        )
        total_steps = self.cfg.epochs * len(loader)
        global_step = 0

        trainable_m = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        print(f"\n MiniVal Trainer | Stage: [{self.cfg.stage.upper()}] | Samples: {len(dataset)} | Trainable: {trainable_m:.2f}M | Device: {self.device}")
        print("-" * 75)

        self.model.train()
        try:
            for epoch in range(self.cfg.epochs):
                start_time = time.time()
                # Akumulasi di GPU tensor - hindari sync .item() per-step
                accum_loss = torch.zeros((), device=self.device)
                tokens_gpu = torch.zeros((), device=self.device, dtype=torch.int64)

                for step, (input_ids, labels) in enumerate(loader, 1):
                    global_step += 1
                    input_ids = input_ids.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    tokens_gpu += (input_ids != self.pad_id).sum().to(tokens_gpu.dtype)

                    # Update LR
                    lr = self._get_lr(global_step, total_steps)
                    for g in self.optimizer.param_groups:
                        g["lr"] = lr

                    # Forward pass dengan autocast
                    with self.autocast_ctx:
                        out = self.model(input_ids, labels=labels)
                        loss = out.loss
                        if out.aux_loss is not None:
                            loss = loss + out.aux_loss
                        loss = loss / self.cfg.grad_accum_steps

                    # Backward pass
                    self.scaler.scale(loss).backward()
                    accum_loss += loss.detach() * self.cfg.grad_accum_steps

                    # Optimizer step (setiap grad_accum_steps)
                    if global_step % self.cfg.grad_accum_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.grad_clip
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)

                    # Log progress - satu-satunya titik sync GPU
                    if step % self.cfg.log_every == 0 or step == len(loader):
                        elapsed = time.time() - start_time
                        tokens_processed = tokens_gpu.item()
                        tok_sec = tokens_processed / max(elapsed, 1e-4)
                        avg_loss = (accum_loss / step).item()
                        ppl = math.exp(min(avg_loss, 20.0))
                        remaining_steps = len(loader) - step
                        eta_min = remaining_steps / max(step / max(elapsed, 1e-4), 1e-4) / 60

                        print(
                            f"[{self.cfg.stage.upper()}] Ep [{epoch+1}/{self.cfg.epochs}] ({step:4d}/{len(loader)}) | "
                            f"Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | "
                            f"{tok_sec:,.0f} tok/s | LR: {lr:.2e} | ETA: {eta_min:.1f}m"
                        )

                    # Periodic checkpoint
                    if global_step % self.cfg.save_every == 0:
                        self.save_checkpoint(f"{self.cfg.stage}_step{global_step}")

                # Flush gradien akumulasi yang tersisa pada akhir epoch
                if global_step % self.cfg.grad_accum_steps != 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                self.save_checkpoint(f"{self.cfg.stage}_epoch{epoch+1}")

            print(f"\n Training [{self.cfg.stage.upper()}] selesai dengan sukses!")

        except KeyboardInterrupt:
            print("\n Interupsi terdeteksi! Menyimpan checkpoint pengaman...")
            self.save_checkpoint(f"{self.cfg.stage}_interrupted")

    def fit_dpo(self, dataset):
        """Mulai pelatihan Direct Preference Optimization (DPO)."""
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )
        total_steps = self.cfg.epochs * len(loader)
        global_step = 0
        beta = 0.1

        print(f"\n MiniVal DPO Alignment | Pairs: {len(dataset)} | Device: {self.device}")
        print("-" * 75)

        self.model.train()
        try:
            for epoch in range(self.cfg.epochs):
                for step, batch in enumerate(loader, 1):
                    global_step += 1
                    x = torch.cat([batch["x_chosen"].to(self.device), batch["x_rejected"].to(self.device)], dim=0)
                    y = torch.cat([batch["y_chosen"].to(self.device), batch["y_rejected"].to(self.device)], dim=0)
                    mask = torch.cat([batch["mask_chosen"].to(self.device), batch["mask_rejected"].to(self.device)], dim=0)

                    lr = self._get_lr(global_step, total_steps)
                    for g in self.optimizer.param_groups:
                        g["lr"] = lr

                    if self.ref_model is None:
                        raise ValueError("fit_dpo membutuhkan ref_model independen. Inisialisasi MiniValTrainer dengan ref_model.")

                    with torch.no_grad():
                        ref_logits = self.ref_model(x).logits
                        ref_logp = F.log_softmax(ref_logits, dim=2).gather(2, y.unsqueeze(2)).squeeze(-1)

                    with self.autocast_ctx:
                        pi_logits = self.model(x).logits
                        pi_logp = F.log_softmax(pi_logits, dim=2).gather(2, y.unsqueeze(2)).squeeze(-1)
                        loss = dpo_loss_fn(ref_logp, pi_logp, mask, beta=beta)

                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    if step % self.cfg.log_every == 0 or step == len(loader):
                        print(f"[DPO] Ep [{epoch+1}/{self.cfg.epochs}] ({step}/{len(loader)}) | Loss: {loss.item():.4f} | LR: {lr:.2e}")

                self.save_checkpoint(f"dpo_epoch{epoch+1}")
            print("\n DPO Alignment selesai!")
        except KeyboardInterrupt:
            print("\n Interupsi terdeteksi! Menyimpan checkpoint pengaman...")
            self.save_checkpoint("dpo_interrupted")


    def fit_grpo(self, dataset, num_generations: int = 4, beta: float = 0.1, max_new_tokens: int = 256):
        """
        GRPO (Group Relative Policy Optimization) + CISPO loss.
        Untuk penalaran (<think>) dan alignment berbasis reward.

        Pipeline:
          1. Rollout: generate G completion per prompt
          2. Score: hitung reward tiap completion
          3. Compute advantage: normalized dalam group
          4. Update policy dengan clipped ratio loss
        """
        from rollout import RolloutEngine, compute_per_token_logps
        from rewards import RewardEngine

        loader = DataLoader(
            dataset, batch_size=self.cfg.batch_size,
            shuffle=True, drop_last=True, pin_memory=("cuda" in self.device)
        )
        total_steps = self.cfg.epochs * len(loader)
        global_step = 0
        epsilon = 0.2

        reward_fn = RewardEngine(use_thinking=True)
        rollout_engine = RolloutEngine(self.model, self.tokenizer, self.device, self.autocast_ctx)

        # Reference model (frozen copy untuk KL divergence)
        import copy
        ref_model = copy.deepcopy(self.model).eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        print(f"\n MiniVal GRPO | Samples: {len(dataset)} | G={num_generations} | Device: {self.device}")
        print("-" * 75)

        self.model.train()
        try:
            for epoch in range(self.cfg.epochs):
                for step, batch in enumerate(loader, 1):
                    global_step += 1
                    # Batch bisa berupa (prompt_ids, attn_mask) atau dict
                    if isinstance(batch, (list, tuple)):
                        prompt_ids, attn_mask = batch[0].to(self.device), batch[1].to(self.device)
                        prompts_text = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
                    else:
                        prompt_ids  = batch["input_ids"].to(self.device)
                        attn_mask   = batch.get("attention_mask", (prompt_ids != self.tokenizer.pad_token_id).long()).to(self.device)
                        prompts_text = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)

                    lr = self._get_lr(global_step, total_steps)
                    for g in self.optimizer.param_groups:
                        g["lr"] = lr

                    # 1. Rollout
                    result = rollout_engine.rollout(
                        prompt_ids, attn_mask,
                        num_generations=num_generations,
                        max_new_tokens=max_new_tokens,
                    )
                    # prompts_text perlu diulang sesuai num_generations
                    rep_prompts = [p for p in prompts_text for _ in range(num_generations)]

                    # 2. Reward scoring
                    rewards = reward_fn.compute(rep_prompts, result.completions, device=self.device)

                    # 3. Group advantage normalization
                    B = len(prompts_text)
                    grouped = rewards.view(B, num_generations)
                    mean_r = grouped.mean(1).repeat_interleave(num_generations)
                    std_r  = grouped.std(1, unbiased=False).repeat_interleave(num_generations)
                    advantages = (rewards - mean_r) / (std_r + 1e-4)

                    # 4. CISPO loss (clamp ratio, tidak clip)
                    with self.autocast_ctx:
                        out = self.model(result.output_ids, attention_mask=(result.output_ids != (self.tokenizer.pad_token_id or 0)).long())
                        aux = out.aux_loss or torch.tensor(0.0, device=self.device)

                    R = result.completion_ids.size(1)
                    P = result.output_ids.size(1) - R
                    logp_pos = (torch.arange(R, device=self.device).unsqueeze(0) + P - 1).expand(result.output_ids.size(0), -1)
                    pi_logp = F.log_softmax(out.logits[:, :-1, :], dim=-1).gather(
                        2, result.output_ids[:, 1:].unsqueeze(-1)
                    ).squeeze(-1).gather(1, logp_pos)

                    with torch.no_grad():
                        ref_out = ref_model(result.output_ids)
                        ref_logp = F.log_softmax(ref_out.logits[:, :-1, :], dim=-1).gather(
                            2, result.output_ids[:, 1:].unsqueeze(-1)
                        ).squeeze(-1).gather(1, logp_pos)

                    mask = result.completion_mask.to(self.device)
                    kl = torch.exp(ref_logp - pi_logp) - (ref_logp - pi_logp) - 1
                    ratio = torch.exp(pi_logp - result.per_token_logps.to(self.device))
                    clip_eps = 0.2
                    surr1 = ratio * advantages.unsqueeze(1)
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages.unsqueeze(1)
                    per_tok_loss = -(torch.min(surr1, surr2) - beta * kl)
                    loss = ((per_tok_loss * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()
                    loss = (loss + aux) / self.cfg.grad_accum_steps

                    self.scaler.scale(loss).backward()
                    if global_step % self.cfg.grad_accum_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)

                    if step % self.cfg.log_every == 0 or step == len(loader):
                        avg_r = rewards.mean().item()
                        print(f"[GRPO] Ep [{epoch+1}/{self.cfg.epochs}] ({step}/{len(loader)}) | "
                              f"Loss: {loss.item()*self.cfg.grad_accum_steps:.4f} | "
                              f"Avg Reward: {avg_r:.3f} | LR: {lr:.2e}")

                self.save_checkpoint(f"grpo_epoch{epoch+1}")
            print("\n GRPO Training selesai!")
        except KeyboardInterrupt:
            print("\n Interupsi! Menyimpan checkpoint...")
            self.save_checkpoint("grpo_interrupted")

    def fit_distill(self, dataset, teacher_model, temperature: float = 2.0):
        """
        Knowledge Distillation: train student model meniru distribusi token teacher.
        KL(student || teacher) dengan temperature scaling.
        """
        loader = DataLoader(
            dataset, batch_size=self.cfg.batch_size,
            shuffle=True, drop_last=True, pin_memory=("cuda" in self.device),
            num_workers=2 if "cuda" in self.device else 0,
            persistent_workers=("cuda" in self.device),
            prefetch_factor=2 if "cuda" in self.device else None,
        )
        total_steps = self.cfg.epochs * len(loader)
        global_step = 0

        teacher_model.to(self.device).eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

        print(f"\n MiniVal Distillation | Samples: {len(dataset)} | T={temperature} | Device: {self.device}")
        print("-" * 75)

        self.model.train()
        try:
            for epoch in range(self.cfg.epochs):
                accum_loss, tokens = 0.0, 0
                start = time.time()

                for step, (input_ids, labels) in enumerate(loader, 1):
                    global_step += 1
                    input_ids = input_ids.to(self.device, non_blocking=True)
                    labels    = labels.to(self.device, non_blocking=True)
                    tokens   += (input_ids != (self.tokenizer.pad_token_id or -1)).sum().item()

                    lr = self._get_lr(global_step, total_steps)
                    for g in self.optimizer.param_groups:
                        g["lr"] = lr

                    with self.autocast_ctx:
                        student_out  = self.model(input_ids, labels=labels)
                        with torch.no_grad():
                            teacher_logits = teacher_model(input_ids).logits

                        # Soft-label KL loss dengan temperature
                        s_logits = student_out.logits[..., :-1, :].contiguous()
                        t_logits = teacher_logits[..., :-1, :].contiguous()
                        t_probs  = F.softmax(t_logits / temperature, dim=-1).detach()
                        s_lprobs = F.log_softmax(s_logits / temperature, dim=-1)
                        kl_loss  = F.kl_div(s_lprobs, t_probs, reduction="batchmean") * (temperature ** 2)

                        # Kombinasi hard-label CE + soft-label KL
                        hard_loss = student_out.loss
                        loss = (0.5 * hard_loss + 0.5 * kl_loss) / self.cfg.grad_accum_steps

                    self.scaler.scale(loss).backward()
                    accum_loss += loss.item() * self.cfg.grad_accum_steps

                    if global_step % self.cfg.grad_accum_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)

                    if step % self.cfg.log_every == 0 or step == len(loader):
                        elapsed = time.time() - start
                        print(f"[DISTILL] Ep [{epoch+1}/{self.cfg.epochs}] ({step:4d}/{len(loader)}) | "
                              f"Loss: {accum_loss/step:.4f} | {tokens/max(elapsed,1e-4):,.0f} tok/s | LR: {lr:.2e}")

                self.save_checkpoint(f"distill_epoch{epoch+1}")
            print("\n Distillation selesai!")
        except KeyboardInterrupt:
            print("\n Interupsi! Menyimpan checkpoint...")
            self.save_checkpoint("distill_interrupted")

    def save_checkpoint(self, name: str):
        """Simpan checkpoint model ke file .pth."""
        save_path = os.path.join(self.cfg.save_dir, f"{name}.pth")
        if self.cfg.stage == "lora":
            save_lora(self.model, save_path)
        else:
            raw = getattr(self.model, "_orig_mod", self.model)
            torch.save({k: v.cpu() for k, v in raw.state_dict().items()}, save_path)
        print(f" Checkpoint tersimpan: {save_path}")

