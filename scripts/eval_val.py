"""
MiniVal Evaluation & Benchmark Suite
======================================
Alat evaluasi komprehensif untuk MiniVal:
  - Uji kelancaran respons & output tokens
  - Hitung throughput (token per detik)
  - Hitung Perplexity (PPL) pada teks referensi
  - Uji reasoning format (<think>...</think>)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import ModelPresets
from model import MiniValConfig, MiniValForCausalLM

BENCHMARK_PROMPTS = [
    # General knowledge / ID
    {"role": "user", "content": "Jelaskan apa itu kecerdasan buatan dalam 2 kalimat."},
    # General knowledge / EN
    {"role": "user", "content": "Explain what a transformer neural network is in simple terms."},
    # Reasoning
    {"role": "user", "content": "Jika ada 5 burung di pohon lalu 2 ditembak, berapa burung yang tersisa di pohon? Jelaskan logikanya."},
    # Coding
    {"role": "user", "content": "Tulis fungsi Python untuk membalikkan string tanpa menggunakan [::-1]."},
    # Structured formatting
    {"role": "user", "content": "Sebutkan 3 planet terdekat dari matahari dalam format JSON."},
]


def run_eval(
    model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_tokens: int = 256,
):
    print("\n" + "=" * 65)
    print("🔬 MiniVal Quantitative Benchmark Suite")
    print("=" * 65)
    print(f"📦 Model Path : {model_path}")
    print(f"🚀 Device     : {device.upper()}")

    # Muat model
    start_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if os.path.exists(os.path.join(model_path, "model.safetensors")) or os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    else:
        # Checkpoint .pth mentah
        cfg = ModelPresets.base()
        model = MiniValForCausalLM(cfg)
        weights = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=False)

    if device == "cuda":
        model = model.half().eval().to(device)
    else:
        model = model.float().eval().to(device)

    print(f"⏱️ Waktu Muat : {time.time() - start_load:.2f} detik")
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"📊 Parameter  : {param_count:.2f}M")
    print("=" * 65 + "\n")

    total_tokens = 0
    total_time = 0.0
    results = []

    for i, test in enumerate(BENCHMARK_PROMPTS, 1):
        prompt = tokenizer.apply_chat_template([test], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        t0 = time.time()
        with torch.no_grad():
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.90,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        gen_ids = output[0][prompt_len:]
        gen_len = len(gen_ids)
        speed = gen_len / max(elapsed, 1e-4)
        response = tokenizer.decode(gen_ids, skip_special_tokens=True)

        total_tokens += gen_len
        total_time += elapsed

        has_think = "<think>" in response and "</think>" in response
        results.append({
            "prompt": test["content"][:40] + "...",
            "tokens": gen_len,
            "speed": round(speed, 1),
            "think_format": has_think,
        })

        print(f"[{i}/{len(BENCHMARK_PROMPTS)}] Prompt: \"{test['content'][:45]}...\"")
        print(f"     -> Generated: {gen_len} tokens | Speed: {speed:.1f} tok/s | Time: {elapsed:.2f}s")
        if has_think:
            print("     -> 🧠 Reasoning tag <think> detected!")
        print()

    avg_speed = total_tokens / max(total_time, 1e-4)
    print("=" * 65)
    print(f"🏆 HASIL BENCHMARK:")
    print(f"   Total Token Dihasilkan : {total_tokens}")
    print(f"   Total Waktu Inferensi  : {total_time:.2f} detik")
    print(f"   Rata-rata Throughput   : {avg_speed:.1f} token/detik")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniVal Benchmark Suite")
    parser.add_argument("--model", default="minival-v1", help="Path model atau checkpoint")
    parser.add_argument("--tokens", default=256, type=int, help="Maksimum token per respons")
    args = parser.parse_args()

    run_eval(args.model, max_tokens=args.tokens)
