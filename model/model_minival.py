"""
MiniVal — Arsitektur Model Bahasa (Language Model Architecture)
================================================================
Mendukung Dense dan Sparse Mixture-of-Experts (MoE).
Kompatibel penuh dengan HuggingFace Transformers & PEFT.

Fitur utama:
  - Grouped Query Attention (GQA) dengan QK-Norm untuk training stability
  - SwiGLU Feed-Forward (Dense dan MoE)
  - RoPE + YaRN (konteks panjang tanpa fine-tuning ulang)
  - KV-Cache untuk inferensi efisien
  - Flash Attention (SDPA) dengan fallback manual yang benar
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


# ─────────────────────────────────────────────────────────────
# Konfigurasi
# ─────────────────────────────────────────────────────────────

class MiniValConfig(PretrainedConfig):
    """
    Konfigurasi arsitektur MiniVal.

    Semua hyperparameter model didefinisikan di sini — tidak ada
    magic number tersebar di seluruh kode.
    """
    model_type = "minival"

    def __init__(
        self,
        # Dimensi inti
        vocab_size: int = 6400,
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        # Attention
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,       # GQA: < num_attention_heads -> aktifkan GQA
        head_dim: Optional[int] = None,     # default: hidden_size // num_attention_heads
        # Feed-Forward
        hidden_act: str = "silu",
        intermediate_size: Optional[int] = None,  # default: ceil(hidden_size * pi / 64) * 64
        # Normalisasi & regularisasi
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        # Posisi & konteks
        max_position_embeddings: int = 32768,
        rope_theta: float = 1_000_000.0,
        inference_rope_scaling: bool = False,  # aktifkan YaRN saat inferensi > max_position_embeddings
        # Embedding
        tie_word_embeddings: bool = True,
        # Attention acceleration
        flash_attn: bool = True,
        # Mixture-of-Experts
        use_moe: bool = False,
        num_experts: int = 4,
        num_experts_per_tok: int = 1,
        moe_intermediate_size: Optional[int] = None,
        norm_topk_prob: bool = True,
        router_aux_loss_coef: float = 5e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)

        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size or (
            math.ceil(hidden_size * math.pi / 64) * 64
        )

        self.rms_norm_eps = rms_norm_eps
        self.dropout = dropout

        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.rope_scaling = _yarn_config() if inference_rope_scaling else None

        self.tie_word_embeddings = tie_word_embeddings
        self.flash_attn = flash_attn

        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size or self.intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef

        # HuggingFace token IDs
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)


def _yarn_config() -> dict:
    """Konfigurasi YaRN default untuk context-length extrapolation."""
    return {
        "type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 2048,
        "beta_fast": 32,
        "beta_slow": 1,
        "attention_factor": 1.0,
    }


# ─────────────────────────────────────────────────────────────
# Komponen Dasar
# ─────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Lebih ringan dari LayerNorm karena tidak menghitung mean.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast ke float32 untuk stabilitas numerik, kemudian kembalikan ke dtype asli
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * normed).to(x.dtype)


# ─────────────────────────────────────────────────────────────
# RoPE (Rotary Position Embedding) + YaRN
# ─────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    RoPE dengan dukungan YaRN untuk extrapolasi konteks panjang.

    Dipisahkan menjadi modul tersendiri agar:
    1. Mudah diuji secara independen
    2. Frekuensi dihitung dan di-cache secara modular
    """
    def __init__(self, config: MiniValConfig):
        super().__init__()
        freqs_cos, freqs_sin = self._compute_freqs(config)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    @staticmethod
    def _compute_freqs(config: MiniValConfig):
        dim = config.head_dim
        end = config.max_position_embeddings
        base = config.rope_theta
        scaling = config.rope_scaling

        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        attn_factor = 1.0

        if scaling is not None:
            orig_max = scaling.get("original_max_position_embeddings", 2048)
            factor = scaling.get("factor", 16)
            beta_fast = scaling.get("beta_fast", 32.0)
            beta_slow = scaling.get("beta_slow", 1.0)
            attn_factor = scaling.get("attention_factor", 1.0)

            if end / orig_max > 1.0:
                def inv_dim(b: float) -> float:
                    return (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(base))

                low = max(math.floor(inv_dim(beta_fast)), 0)
                high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
                ramp = torch.clamp(
                    (torch.arange(dim // 2).float() - low) / max(high - low, 0.001),
                    0, 1,
                )
                freqs = freqs * (1 - ramp + ramp / factor)

        t = torch.arange(end)
        outer = torch.outer(t, freqs).float()
        freqs_cos = torch.cat([outer.cos(), outer.cos()], dim=-1) * attn_factor
        freqs_sin = torch.cat([outer.sin(), outer.sin()], dim=-1) * attn_factor
        return freqs_cos, freqs_sin

    def forward(self, seq_len: int, start_pos: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Kembalikan (cos, sin) slice untuk posisi [start_pos, start_pos + seq_len)."""
        return (
            self.freqs_cos[start_pos: start_pos + seq_len],
            self.freqs_sin[start_pos: start_pos + seq_len],
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Terapkan RoPE ke query dan key. Input shape: (B, S, H, D)."""
    cos = cos.unsqueeze(1)  # (S, 1, D) -> broadcast ke (B, S, H, D)
    sin = sin.unsqueeze(1)
    q_out = (q * cos + _rotate_half(q) * sin).to(q.dtype)
    k_out = (k * cos + _rotate_half(k) * sin).to(k.dtype)
    return q_out, k_out


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Ekspansi KV head untuk GQA. Tidak copy data jika n_rep == 1."""
    if n_rep == 1:
        return x
    B, S, H, D = x.shape
    return x[:, :, :, None, :].expand(B, S, H, n_rep, D).reshape(B, S, H * n_rep, D)


# ─────────────────────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """
    Grouped Query Attention (GQA) dengan:
    - QK-Norm: stabilisasi training untuk model besar
    - KV-Cache: inferensi efisien (O(1) per token)
    - Flash Attention (SDPA) dengan fallback manual yang benar
    """
    def __init__(self, config: MiniValConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # faktor ekspansi GQA
        self.head_dim = config.head_dim
        self.dropout_p = config.dropout

        # Projeksi Q, K, V, O — tanpa bias (standar modern LLM)
        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)

        # QK-Norm: normalisasi sebelum RoPE mencegah attention collapse
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # Flash attn hanya aman saat prefill (bukan decode dengan KV cache)
        self._use_flash = config.flash_attn and hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        x: torch.Tensor,                              # (B, S, D)
        pos_emb: tuple[torch.Tensor, torch.Tensor],   # (cos, sin)
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        # QK-Norm sebelum RoPE
        q, k = self.q_norm(q), self.k_norm(k)

        # RoPE
        cos, sin = pos_emb
        q, k = apply_rotary_emb(q, k, cos, sin)

        # KV-Cache: concatenate dengan token lama
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=1)
            v = torch.cat([past_kv[1], v], dim=1)
        next_kv = (k, v) if use_cache else None

        # Siapkan untuk attention: (B, H, S, D)
        q = q.transpose(1, 2)
        k = repeat_kv(k, self.n_rep).transpose(1, 2)
        v = repeat_kv(v, self.n_rep).transpose(1, 2)

        # Flash Attention: hanya saat prefill (S > 1) dan tanpa KV cache aktif
        is_prefill = S > 1 and past_kv is None
        if self._use_flash and is_prefill:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual attention dengan causal mask yang benar
            scale = math.sqrt(self.head_dim)
            scores = (q @ k.transpose(-2, -1)) / scale
            total_len = k.shape[-2]
            mask = torch.full((S, total_len), float("-inf"), device=scores.device, dtype=scores.dtype)
            mask = mask.triu(total_len - S + 1)
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
            out = self.attn_drop(F.softmax(scores.float(), dim=-1).to(q.dtype)) @ v

        out = out.transpose(1, 2).reshape(B, S, -1)
        out = self.resid_drop(self.o_proj(out))
        return out, next_kv


# ─────────────────────────────────────────────────────────────
# Feed-Forward
# ─────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    FFN(x) = down(silu(gate(x)) * up(x))
    """
    def __init__(self, config: MiniValConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        d_ff = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, d_ff, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, config.hidden_size, bias=False)
        self.act_fn    = ACT2FN[config.hidden_act]
        self.drop      = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEFeedForward(nn.Module):
    """
    Sparse Mixture-of-Experts FFN.

    Setiap token hanya melewati top-k expert (default k=1).
    Router menggunakan softmax + load-balancing auxiliary loss.
    """
    def __init__(self, config: MiniValConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk = config.norm_topk_prob
        self.aux_coef = config.router_aux_loss_coef

        # Router gate: linear projection ke expert scores
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        flat = x.view(-1, D)  # (B*S, D)

        # Routing scores
        scores = F.softmax(self.gate(flat), dim=-1)  # (B*S, E)
        topk_w, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.norm_topk and self.top_k > 1:
            topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-20)

        # Dispatch: setiap token ke expert yang terpilih
        out = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i).any(dim=-1)
            if mask.any():
                tok_idx = mask.nonzero(as_tuple=True)[0]
                w = topk_w[tok_idx, (topk_idx[tok_idx] == i).nonzero(as_tuple=True)[1]]
                out[tok_idx] += expert(flat[tok_idx]) * w.unsqueeze(-1)
            elif self.training:
                out[0] = out[0] + 0.0 * sum(p.sum() for p in expert.parameters())

        # Load-balancing auxiliary loss
        if self.training and self.aux_coef > 0:
            load = F.one_hot(topk_idx, self.num_experts).float().mean(dim=(0, 1))
            prob = scores.mean(dim=0)
            aux_loss = (load * prob).sum() * self.num_experts * self.aux_coef
        else:
            aux_loss = scores.new_zeros(1).squeeze()

        return out.view(B, S, D), aux_loss


# ─────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────

class MiniValBlock(nn.Module):
    """
    Satu layer decoder Transformer.
    Nama atribut mengikuti standar model.safetensors agar bobot ter-load utuh:
      self_attn, mlp, input_layernorm, post_attention_layernorm
    """
    def __init__(self, config: MiniValConfig):
        super().__init__()
        self.self_attn                = Attention(config)
        self.mlp                      = MoEFeedForward(config) if config.use_moe else FeedForward(config)
        self.input_layernorm          = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.use_moe                  = config.use_moe

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: tuple[torch.Tensor, torch.Tensor],
        past_kv: Optional[tuple] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple], Optional[torch.Tensor]]:
        attn_out, next_kv = self.self_attn(self.input_layernorm(x), pos_emb, past_kv, use_cache)
        x = x + attn_out

        if self.use_moe:
            ffn_out, aux_loss = self.mlp(self.post_attention_layernorm(x))
        else:
            ffn_out = self.mlp(self.post_attention_layernorm(x))
            aux_loss = None
        x = x + ffn_out

        return x, next_kv, aux_loss


# ─────────────────────────────────────────────────────────────
# Model Utama
# ─────────────────────────────────────────────────────────────

class MiniValModel(PreTrainedModel):
    """
    Stack transformer decoder MiniVal.
    Menghasilkan hidden states, past KV caches, dan aux loss (MoE).
    """
    config_class = MiniValConfig

    def __init__(self, config: MiniValConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.drop         = nn.Dropout(config.dropout)
        self.layers       = nn.ModuleList([MiniValBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm         = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb   = RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list, Optional[torch.Tensor]]:
        B, S = input_ids.shape

        h = self.drop(self.embed_tokens(input_ids))

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        # Guard: handle DynamicCache dari HuggingFace generate() atau list [None, None, ...]
        if hasattr(past_key_values, "layers"):
            past_key_values = [None] * len(self.layers)

        first_kv = past_key_values[0] if past_key_values else None
        start_pos = first_kv[0].shape[1] if (first_kv is not None and isinstance(first_kv, (tuple, list))) else 0
        pos_emb = self.rotary_emb(S, start_pos)

        next_kvs: list = []
        total_aux: Optional[torch.Tensor] = None

        for layer, past_kv in zip(self.layers, past_key_values):
            h, next_kv, aux = layer(h, pos_emb, past_kv, use_cache)
            next_kvs.append(next_kv)
            if aux is not None:
                total_aux = aux if total_aux is None else total_aux + aux

        h = self.norm(h)
        return h, next_kvs, total_aux


class MiniValForCausalLM(PreTrainedModel, GenerationMixin):
    """
    MiniVal Causal Language Model — siap untuk Pretraining, SFT, dan Inferensi.
    """
    config_class = MiniValConfig
    _tied_weights_keys = ["lm_head.weight", "model.embed_tokens.weight"]

    def __init__(self, config: MiniValConfig):
        super().__init__(config)
        self.model   = MiniValModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> MoeCausalLMOutputWithPast:
        hidden, past_kvs, aux_loss = self.model(input_ids, past_key_values, use_cache)

        h = hidden[:, -logits_to_keep:, :] if logits_to_keep > 0 else hidden
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=past_kvs,
            hidden_states=hidden,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[list] = None,
        **kwargs,
    ) -> dict:
        first_kv = past_key_values[0] if (isinstance(past_key_values, (list, tuple)) and past_key_values) else None
        if first_kv is not None and isinstance(first_kv, (list, tuple)) and first_kv[0] is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }
