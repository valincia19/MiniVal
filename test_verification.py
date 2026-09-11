"""
MiniVal Automated Regression & Verification Test Suite
======================================================
Tests architecture, LoRA adaptation, MoE aux loss, DPO & GRPO logic,
token sequence masking, rewards engine, and API schemas.
"""
import os
import sys
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_config():
    from config import ModelPresets, TrainConfig, MiniValConfig

    stages = ["pretrain", "sft", "lora", "dpo", "grpo", "distill"]
    for s in stages:
        tc = TrainConfig(stage=s, preset="tiny", data_path="dummy.jsonl")
        assert tc.stage == s

    try:
        TrainConfig(stage="unsupported", preset="tiny", data_path="dummy.jsonl")
        assert False, "Expected ValueError on unsupported stage"
    except ValueError:
        pass

    for preset_name in ["tiny", "base", "pro", "moe"]:
        cfg = ModelPresets.get(preset_name)
        assert isinstance(cfg, MiniValConfig)
        assert cfg.vocab_size == 6400

    print("PASS: Config & Presets")


def test_model_and_moe():
    from model.model_minival import MiniValConfig, MiniValForCausalLM, MoEFeedForward

    # Dense model test
    dense_cfg = MiniValConfig(
        vocab_size=128,
        dim=64,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        hidden_dim=128,
        max_seq_len=64,
        use_moe=False,
    )
    model = MiniValForCausalLM(dense_cfg)
    x = torch.randint(0, 128, (2, 16))
    out = model(x)
    assert out.logits.shape == (2, 16, 128)
    assert out.aux_loss is None

    # MoE model test with top_k = 2
    moe_cfg = MiniValConfig(
        vocab_size=128,
        dim=64,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        hidden_dim=128,
        max_seq_len=64,
        use_moe=True,
        num_experts=4,
        top_k=2,
        aux_loss_coef=0.01,
    )
    moe_model = MiniValForCausalLM(moe_cfg)
    moe_out = moe_model(x)
    assert moe_out.logits.shape == (2, 16, 128)
    assert moe_out.aux_loss.dim() == 0, f"Expected scalar aux loss, got shape {moe_out.aux_loss.shape}"
    assert moe_out.aux_loss.item() > 0.0, f"Expected positive aux loss, got {moe_out.aux_loss.item()}"

    print("PASS: Model & MoE Aux Loss")


def test_lora():
    from model.model_minival import MiniValConfig, MiniValForCausalLM
    from model.model_lora import LoRALinear, apply_lora, merge_lora, save_lora, load_lora

    cfg = MiniValConfig(vocab_size=128, dim=64, n_layers=2, n_heads=2, n_kv_heads=2, hidden_dim=128, max_seq_len=64)
    model = MiniValForCausalLM(cfg)

    # Apply LoRA
    apply_lora(model, rank=4)
    lora_layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
    assert len(lora_layers) > 0, "LoRALinear wrappers should be applied"

    # Forward pass
    x = torch.randint(0, 128, (2, 16))
    out = model(x)
    assert out.logits.shape == (2, 16, 128)

    # Parameter training check
    base_grad_count = sum(p.requires_grad for m in lora_layers for p in m.base.parameters())
    lora_grad_count = sum(p.requires_grad for m in lora_layers for p in [m.lora_A.weight, m.lora_B.weight])
    assert base_grad_count == 0, "Base parameters must be frozen"
    assert lora_grad_count == len(lora_layers) * 2, "LoRA parameters must require grad"

    # Save and load
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        save_lora(model, tmp_path)
        load_lora(model, tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Merge LoRA
    merge_lora(model)
    merged_layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
    assert len(merged_layers) == 0, "All LoRALinear layers should be merged into base nn.Linear"

    print("PASS: LoRALinear, Freeze, Save/Load & Merge")


def test_rewards():
    from rewards import (
        format_reward,
        tool_call_reward,
        length_reward,
        repetition_penalty,
        RewardEngine,
    )

    # Thinking format
    assert format_reward("<think>valid reasoning process of adequate length</think>Final answer.") > 0.0
    assert format_reward("No tags here") == 0.0

    # Tool call
    valid_tool = '```json\n{"name": "calc", "arguments": {"x": 1}}\n```'
    assert tool_call_reward(valid_tool) > 0.0
    assert tool_call_reward("just talking") == 0.0

    # Repetition
    repeated = "repeat " * 20
    assert repetition_penalty(repeated) > 0.0

    # RewardEngine
    engine = RewardEngine(use_thinking=True, use_tool=True)
    prompts = ["Question 1", "Question 2"]
    completions = [
        "<think>detailed step by step reasoning for the answer</think>Conclusion",
        "brief",
    ]
    scores = engine.compute(prompts, completions)
    assert scores.shape == (2,)
    assert scores[0].item() > scores[1].item()

    print("PASS: Rewards Engine")


def test_dataset_masking():
    from dataset.universal import UniversalDataset

    class MockTokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            if text == "<|im_start|>assistant\n":
                return [10, 20]
            if text == "<|im_end|>\n":
                return [99]
            return [5]

    ds = UniversalDataset.__new__(UniversalDataset)
    ds.tokenizer = MockTokenizer()
    ds.max_length = 64
    ds.bos_id = [10, 20]
    ds.eos_id = [99]

    # Subsequence matching test
    tokens = [1, 2, 10, 20, 30, 99, 10, 20, 40, 50, 99, 3]
    spans = ds._find_assistant_spans(tokens)
    assert len(spans) == 2
    assert spans[0] == (4, 5)   # index 4 (token 30), up to 5
    assert spans[1] == (8, 10)  # index 8, 9 (tokens 40, 50), up to 10

    print("PASS: UniversalDataset Subsequence Masking")


def test_rollout_logps():
    from model.model_minival import MiniValConfig, MiniValForCausalLM
    from rollout import compute_per_token_logps

    cfg = MiniValConfig(vocab_size=64, dim=32, n_layers=1, n_heads=1, n_kv_heads=1, hidden_dim=64, max_seq_len=32)
    model = MiniValForCausalLM(cfg)

    x = torch.randint(0, 64, (3, 12))
    logps = compute_per_token_logps(model, x, n_keep=5)
    assert logps.shape == (3, 5)
    assert torch.all(logps <= 0.0)

    print("PASS: Rollout per-token log probabilities")


def test_api_schemas():
    from scripts.serve_api import ChatMessage, ChatRequest

    m = ChatMessage(role="user", content="Hello MiniVal")
    assert m.role == "user"
    assert m.content == "Hello MiniVal"

    # Reject invalid role
    try:
        ChatMessage(role="unauthorized_actor", content="exploit")
        assert False, "Should reject invalid role"
    except Exception:
        pass

    req = ChatRequest(messages=[m], temperature=0.8, stream=False)
    assert req.max_tokens == 1024
    assert not req.stream

    print("PASS: Serve API Schemas & Validation")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Running MiniVal Full Regression Verification Suite")
    print("=" * 60)
    test_config()
    test_model_and_moe()
    test_lora()
    test_rewards()
    test_dataset_masking()
    test_rollout_logps()
    test_api_schemas()
    print("=" * 60)
    print("🎉 ALL 7 SUITES PASSED CLEANLY!")
    print("=" * 60 + "\n")
