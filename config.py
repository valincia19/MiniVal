"""
MiniVal - Konfigurasi & Preset Model
======================================
Satu tempat untuk semua preset arsitektur dan training config.
Tidak perlu hafal argumen -- cukup pilih nama preset.
"""
from __future__ import annotations

from dataclasses import dataclass

from model.model_minival import MiniValConfig


# Preset Arsitektur

@dataclass
class ModelPresets:
    """Akses preset model via ModelPresets.get("base") atau ModelPresets.base()."""

    @staticmethod
    def tiny() -> MiniValConfig:
        """26M Parameters - super cepat untuk eksperimen lokal & debugging."""
        return MiniValConfig(
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            num_key_value_heads=2,
            max_position_embeddings=2048,
            vocab_size=6400,
        )

    @staticmethod
    def base() -> MiniValConfig:
        """108M Parameters - standar untuk pretraining & conversational AI."""
        return MiniValConfig(
            hidden_size=768,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=4096,
            vocab_size=6400,
        )

    @staticmethod
    def pro() -> MiniValConfig:
        """~300M Parameters - konfigurasi frontier-grade untuk performa produksi maksimal."""
        return MiniValConfig(
            hidden_size=1024,
            num_hidden_layers=14,
            num_attention_heads=16,
            num_key_value_heads=4,
            intermediate_size=2816,
            max_position_embeddings=8192,
            vocab_size=6400,
            inference_rope_scaling=True,
        )

    @staticmethod
    def moe() -> MiniValConfig:
        """MoE (~108M aktif) - hemat komputasi dengan 4 routed experts."""
        return MiniValConfig(
            hidden_size=768,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            use_moe=True,
            num_experts=4,
            num_experts_per_tok=1,
            max_position_embeddings=4096,
            vocab_size=6400,
        )

    @classmethod
    def get(cls, name: str) -> MiniValConfig:
        """Ambil konfigurasi berdasarkan nama preset ('tiny', 'base', 'pro', 'moe')."""
        presets = {
            "tiny": cls.tiny,
            "base": cls.base,
            "pro": cls.pro,
            "moe": cls.moe,
        }
        key = str(name).lower()
        if key not in presets:
            valid = ", ".join(presets)
            raise ValueError(f"Preset '{name}' tidak dikenal. Pilihan yang valid: {valid}")
        return presets[key]()

    @classmethod
    def list(cls) -> dict[str, str]:
        """Daftar semua preset dengan deskripsinya."""
        return {
            "tiny": "26M params - cepat untuk debugging & laptop tanpa GPU",
            "base": "108M params - standar untuk dialog & teks umum",
            "pro": "300M params - konfigurasi frontier (GQA 4x, YaRN 8k-32k, 14 layers)",
            "moe": "108M params aktif - Mixture of Experts (4 experts, top-1)",
        }


# Konfigurasi Training

@dataclass
class TrainConfig:
    """Konfigurasi terpadu untuk pipeline pelatihan MiniVal."""
    stage: str = "sft"               # 'pretrain', 'sft', 'lora', 'dpo', 'grpo', atau 'distill'
    preset: str = "base"             # 'tiny', 'base', atau 'moe'
    data_path: str = "dataset/sample_data.jsonl"
    save_dir: str = "out"
    resume_from: str = "none"        # path ke checkpoint .pth atau 'none'

    # Loop
    epochs: int = 2
    batch_size: int = 16
    grad_accum_steps: int = 4        # batch efektif = batch_size * grad_accum_steps
    max_seq_len: int = 1024

    # Optimizer & Scheduler
    lr: float = 2e-5
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    warmup_ratio: float = 0.05       # proporsi total steps untuk linear warmup

    # LoRA (khusus stage='lora')
    lora_rank: int = 16

    # Logging & Checkpoint
    log_every: int = 10
    save_every: int = 200

    def __post_init__(self):
        valid_stages = {"pretrain", "sft", "lora", "dpo", "grpo", "distill"}
        if self.stage not in valid_stages:
            raise ValueError(f"stage harus salah satu dari {valid_stages}, bukan '{self.stage}'")
