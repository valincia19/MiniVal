"""
MiniVal Rollout Engine
=======================
Mesin inferensi batch cepat untuk training Reinforcement Learning (GRPO / PPO).
Menghasilkan beberapa completion sekaligus dari satu prompt, beserta per-token logprob.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel


@dataclass
class RolloutResult:
    """
    Hasil satu batch rollout.

    Shapes (B=batch, G=num_generations, P=prompt_len, R=completion_len):
      output_ids      : (B*G, P+R)
      completion_ids  : (B*G, R)
      per_token_logps : (B*G, R)
      completions     : list[str]
      prompt_lens     : (B*G,)
      completion_mask : (B*G, R)  -- 1=real, 0=pad
    """
    output_ids:      Tensor
    completion_ids:  Tensor
    per_token_logps: Tensor
    completions:     list[str]
    prompt_lens:     Tensor
    completion_mask: Tensor


def compute_per_token_logps(
    model: torch.nn.Module,
    input_ids: Tensor,
    n_keep: int,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """Log-prob per token untuk n_keep token terakhir. Returns (B, n_keep)."""
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    logits = raw(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    target = input_ids[:, -n_keep:]
    return F.log_softmax(logits, dim=-1).gather(2, target.unsqueeze(-1)).squeeze(-1)


class RolloutEngine:
    """
    Rollout engine: generate completions + per-token logprob untuk RL training.

    Contoh:
        engine = RolloutEngine(model, tokenizer, device="cuda")
        result = engine.rollout(prompt_ids, attn_mask, num_generations=6, max_new_tokens=512)
    """

    def __init__(self, model, tokenizer, device: str = "cuda", autocast_ctx=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._ctx = autocast_ctx or nullcontext()

    def update_policy(self, model):
        """Update referensi model (misal setelah DDP wrap atau model swapping)."""
        self.model = model

    @torch.no_grad()
    def rollout(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor,
        num_generations: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> RolloutResult:
        raw = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        eos_id = self.tokenizer.eos_token_id

        rep_ids  = prompt_ids.repeat_interleave(num_generations, dim=0)
        rep_mask = attention_mask.repeat_interleave(num_generations, dim=0)
        prompt_lens = rep_mask.sum(dim=1)

        with self._ctx:
            output_ids = raw.generate(
                input_ids=rep_ids,
                attention_mask=rep_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pad_id,
            )

        P = rep_ids.size(1)
        completion_ids = output_ids[:, P:]
        R = completion_ids.size(1)

        is_eos = (completion_ids == eos_id)
        has_eos = is_eos.any(dim=1)
        eos_pos = is_eos.int().argmax(dim=1)
        idx = torch.arange(R, device=self.device).unsqueeze(0)
        completion_mask = (idx <= eos_pos.unsqueeze(1)).float()
        completion_mask[~has_eos] = (completion_ids[~has_eos] != pad_id).float()

        per_token_logps = compute_per_token_logps(
            raw, output_ids, n_keep=R,
            attention_mask=(output_ids != pad_id).long()
        )
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        return RolloutResult(
            output_ids=output_ids,
            completion_ids=completion_ids,
            per_token_logps=per_token_logps,
            completions=completions,
            prompt_lens=prompt_lens,
            completion_mask=completion_mask,
        )
