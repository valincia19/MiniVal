"""
MiniVal LoRA (Low-Rank Adaptation)
Implementasi LoRA modular tanpa dependensi eksternal.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Linear layer wrapper dengan Low-Rank Adaptation."""

    def __init__(self, base: nn.Linear, rank: int = 16):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(x))


LoRA = LoRALinear


def apply_lora(model: nn.Module, rank: int = 16):
    """Pasang LoRA adapter pada semua proyeksi Linear model."""
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear):
            parts = name.rsplit(".", 1)
            parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
            child_name = parts[-1]
            wrapper = LoRALinear(module, rank=rank).to(module.weight.device)
            setattr(parent, child_name, wrapper)


def load_lora(model: nn.Module, path: str):
    """Muat bobot LoRA adapter dari checkpoint."""
    device = next(model.parameters()).device
    state_dict = torch.load(path, map_location=device, weights_only=True)
    state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}

    raw_model = getattr(model, "_orig_mod", model)
    for name, module in raw_model.named_modules():
        if isinstance(module, LoRALinear):
            clean_name = name[7:] if name.startswith("module.") else name
            a_key = f"{clean_name}.lora_A.weight"
            b_key = f"{clean_name}.lora_B.weight"
            if a_key in state_dict:
                module.lora_A.weight.data.copy_(state_dict[a_key])
            if b_key in state_dict:
                module.lora_B.weight.data.copy_(state_dict[b_key])


def save_lora(model: nn.Module, path: str):
    """Simpan parameter LoRA ke file checkpoint."""
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if isinstance(module, LoRALinear):
            clean_name = name[7:] if name.startswith("module.") else name
            state_dict[f"{clean_name}.lora_A.weight"] = module.lora_A.weight.data.cpu()
            state_dict[f"{clean_name}.lora_B.weight"] = module.lora_B.weight.data.cpu()
    torch.save(state_dict, path)


def merge_lora(model: nn.Module, lora_path: Optional[str] = None, save_path: Optional[str] = None):
    """Gabungkan bobot LoRA secara permanen ke bobot model dasar."""
    if lora_path is not None:
        load_lora(model, lora_path)
    raw_model = getattr(model, "_orig_mod", model)
    for name, module in list(raw_model.named_modules()):
        if isinstance(module, LoRALinear):
            delta = module.lora_B.weight.data @ module.lora_A.weight.data
            module.base.weight.data.add_(delta)
            parts = name.rsplit(".", 1)
            parent = raw_model.get_submodule(parts[0]) if len(parts) > 1 else raw_model
            setattr(parent, parts[-1], module.base)
    if save_path is not None:
        torch.save(raw_model.state_dict(), save_path)
