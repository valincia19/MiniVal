"""
MiniVal Reward Engine
======================
Fungsi reward modular untuk Reinforcement Learning (GRPO & PPO).

Komponen reward:
  1. format_reward     : Evaluasi tag <think>...</think> untuk reasoning
  2. length_reward     : Mencegah output degeneratif (terlalu pendek atau rambling)
  3. repetition_penalty: Penalti pengulangan n-gram
  4. tool_call_reward  : Evaluasi integritas skema JSON function calling
"""
from __future__ import annotations

import json
import re
from typing import Optional

import torch

TOOL_CALL_BONUS = 1.5
TOOL_FORMAT_REWARD = 1.0
REPETITION_SCALE = 2.0


def repetition_penalty(text: str, n: int = 3, cap: float = 0.5) -> float:
    """Hitung penalti pengulangan n-gram bounded antara 0.0 hingga cap."""
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    unique_ratio = len(set(ngrams)) / len(ngrams)
    return min(cap, (1.0 - unique_ratio) * REPETITION_SCALE * cap)


def format_reward(response: str, min_think: int = 20, max_think: int = 400) -> float:
    """Evaluasi struktur reasoning tag <think>...</think>."""
    if "</think>" not in response:
        return 0.0

    if response.count("</think>") != 1:
        return -0.5

    parts = response.split("</think>", 1)
    think_text = parts[0].replace("<think>", "").strip()
    ans_text = parts[1].strip()

    if not ans_text:
        return -0.5

    if min_think <= len(think_text) <= max_think:
        return 1.0
    return 0.25


def length_reward(text: str, min_len: int = 15, max_len: int = 1200) -> float:
    """Reward rentang panjang respons yang valid."""
    length = len(text.strip())
    if min_len <= length <= max_len:
        return 0.5
    return -0.5


def tool_call_reward(response: str, expected_tools: Optional[list[str]] = None) -> float:
    """Evaluasi validitas payload JSON function call."""
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
    if not json_match:
        json_match = re.search(r"(\{\"name\"\s*:.*?\})", response, re.DOTALL)

    if not json_match:
        return 0.0

    try:
        data = json.loads(json_match.group(1))
        if "name" in data and ("parameters" in data or "arguments" in data):
            if expected_tools and data["name"] in expected_tools:
                return TOOL_CALL_BONUS
            return TOOL_FORMAT_REWARD
    except json.JSONDecodeError:
        return -0.5

    return 0.0


class RewardEngine:
    """Koleksi skoring terpadu untuk batch completions."""

    def __init__(
        self,
        use_thinking: bool = True,
        use_tool: bool = False,
        reward_model=None,
    ):
        self.use_thinking = use_thinking
        self.use_tool = use_tool
        self.reward_model = reward_model

    def compute(
        self,
        prompts: list[str],
        completions: list[str],
        device: str = "cpu",
    ) -> torch.Tensor:
        """Hitung tensor reward untuk sekumpulan respons."""
        scores = []
        for prompt, resp in zip(prompts, completions):
            score = 0.0

            if self.use_thinking:
                score += format_reward(resp)

            clean_resp = resp.split("</think>")[-1] if "</think>" in resp else resp
            score += length_reward(clean_resp)
            score -= repetition_penalty(clean_resp)

            if self.use_tool:
                score += tool_call_reward(resp)

            scores.append(score)

        return torch.tensor(scores, dtype=torch.float32, device=device)
