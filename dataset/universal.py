"""
MiniVal Universal Dataset Loader
Pemuat data terpadu untuk:
- Pretraining: teks mentah (.txt atau .jsonl dengan field 'text')
- Supervised Fine-Tuning (SFT): dialog multi-turn ('conversations')
- Direct Preference Optimization (DPO): pasangan preferensi ('chosen' & 'rejected')
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

SYSTEM_PROMPTS = [
    "You are MiniVal, a helpful, honest, and harmless AI assistant.",
    "Kamu adalah MiniVal, asisten AI cerdas dan ramah yang siap membantu menjawab pertanyaan.",
    "You are MiniVal, an efficient and precise language model.",
    "Kamu adalah MiniVal, model bahasa yang cepat, tepat, dan terstruktur.",
]


class PadCollate:
    """Collate dynamic: pad batch ke panjang terpanjang di batch (bukan max_length global)."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor]]):
        ids, labs = zip(*batch)
        max_len = max(t.size(0) for t in ids)
        input_ids = torch.full((len(ids), max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((len(ids), max_len), -100, dtype=torch.long)
        for i, (t, l) in enumerate(zip(ids, labs)):
            input_ids[i, : t.size(0)] = t
            labels[i, : l.size(0)] = l
        return input_ids, labels


class UniversalDataset(Dataset):
    """Dataset universal dengan deteksi format otomatis untuk pretrain, SFT, dan DPO."""

    def __init__(self, file_path: str, tokenizer, max_length: int = 1024, max_samples: Optional[int] = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[dict[str, Any]] = []
        self.data_type = "pretrain"

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File dataset tidak ditemukan: {file_path}")

        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                self.samples = [{"text": line.strip()} for line in f if len(line.strip()) > 5]
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not self.samples:
            raise ValueError(f"Dataset kosong atau tidak valid: {file_path}")

        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]

        first = self.samples[0]
        if "chosen" in first and "rejected" in first:
            self.data_type = "dpo"
        elif "conversations" in first:
            self.data_type = "sft"
        else:
            self.data_type = "pretrain"

        self.bos_id = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.eos_id = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)

    def __len__(self) -> int:
        return len(self.samples)

    def get_collate(self) -> PadCollate:
        """Collate fn untuk dynamic padding. Hanya untuk format (input_ids, labels)."""
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None) or 0
        return PadCollate(pad_id)

    def _encode_text(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(text, add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        bos_id = getattr(self.tokenizer, "bos_token_id", None) or 1
        eos_id = getattr(self.tokenizer, "eos_token_id", None) or 2

        # Panjang variabel - pad ditangani PadCollate per-batch
        tokens = [bos_id] + tokens + [eos_id]
        input_ids = tokens[: self.max_length]
        labels = list(input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def _find_assistant_spans(self, input_ids: list[int]) -> list[tuple[int, int]]:
        """Cari rentang indeks token asisten untuk loss masking."""
        spans = []
        i = 0
        bos_len = len(self.bos_id)
        eos_len = len(self.eos_id)
        while i < len(input_ids):
            if input_ids[i : i + bos_len] == self.bos_id:
                start = i + bos_len
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + eos_len] == self.eos_id:
                        break
                    end += 1
                spans.append((start, min(end, self.max_length)))
                i = end + eos_len
            else:
                i += 1
        return spans

    def _encode_chat(self, conversations: list[dict[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
        def norm_role(m: dict) -> str:
            r = str(m.get("role") or m.get("from") or "user").lower()
            if r in ("human", "user"):
                return "user"
            if r in ("gpt", "assistant", "bot", "model"):
                return "assistant"
            if r in ("system",):
                return "system"
            return "user"

        def norm_content(m: dict) -> str:
            return str(m.get("content") or m.get("value") or m.get("text") or "").strip()

        messages = [{"role": norm_role(m), "content": norm_content(m)} for m in conversations]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        input_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids[: self.max_length]

        labels = [-100] * len(input_ids)
        spans = self._find_assistant_spans(input_ids)
        if spans:
            for start, end in spans:
                for j in range(start, end):
                    labels[j] = input_ids[j]
        else:
            for j in range(len(input_ids)):
                labels[j] = input_ids[j]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def _encode_dpo(self, chosen: list[dict[str, str]], rejected: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        prompt_chosen = self.tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
        prompt_rejected = self.tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)
        enc_chosen = self.tokenizer(prompt_chosen, add_special_tokens=False, truncation=True, max_length=self.max_length, padding="max_length")
        enc_rejected = self.tokenizer(prompt_rejected, add_special_tokens=False, truncation=True, max_length=self.max_length, padding="max_length")

        def build_loss_mask(token_ids: list[int]) -> list[int]:
            mask = [0] * len(token_ids)
            for start, end in self._find_assistant_spans(token_ids):
                for j in range(start, end):
                    mask[j] = 1
            return mask

        return {
            "x_chosen": torch.tensor(enc_chosen["input_ids"][:-1], dtype=torch.long),
            "y_chosen": torch.tensor(enc_chosen["input_ids"][1:], dtype=torch.long),
            "mask_chosen": torch.tensor(build_loss_mask(enc_chosen["input_ids"])[1:], dtype=torch.long),
            "x_rejected": torch.tensor(enc_rejected["input_ids"][:-1], dtype=torch.long),
            "y_rejected": torch.tensor(enc_rejected["input_ids"][1:], dtype=torch.long),
            "mask_rejected": torch.tensor(build_loss_mask(enc_rejected["input_ids"])[1:], dtype=torch.long),
        }

    def __getitem__(self, index: int):
        sample = self.samples[index]
        if self.data_type == "dpo":
            return self._encode_dpo(sample["chosen"], sample["rejected"])
        if self.data_type == "sft":
            return self._encode_chat(sample["conversations"])
        return self._encode_text(str(sample.get("text", "")))
