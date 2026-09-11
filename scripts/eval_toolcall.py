"""
MiniVal Tool-Calling & Function Evaluation Benchmark
=====================================================
Uji kemampuan model untuk:
  1. Memilih tool yang tepat berdasarkan pertanyaan
  2. Menghasilkan argumen JSON yang valid
  3. Mematuhi skema fungsi
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

AVAILABLE_TOOLS = [
    {
        "name": "calculate_math",
        "description": "Hitung ekspresi matematika sederhana (+, -, *, /, **).",
        "parameters": {"expression": "string, contoh: 25 * 40"},
    },
    {
        "name": "get_weather",
        "description": "Dapatkan prakiraan cuaca terkini untuk suatu kota.",
        "parameters": {"city": "string, contoh: Jakarta"},
    },
    {
        "name": "currency_converter",
        "description": "Konversi nilai mata uang asing.",
        "parameters": {"amount": "number", "from_cur": "string", "to_cur": "string"},
    },
]

TEST_CASES = [
    {
        "prompt": "Berapa hasil dari 125 dikali 8?",
        "expected_tool": "calculate_math",
    },
    {
        "prompt": "Bagaimana cuaca di kota Surabaya hari ini?",
        "expected_tool": "get_weather",
    },
    {
        "prompt": "Tolong ubah 50 dollar USD ke IDR rupiah.",
        "expected_tool": "currency_converter",
    },
]


def extract_tool_call(text: str) -> dict | None:
    """Ekstrak panggilan tool JSON dari respons model."""
    # Format ```json { ... } ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        m = re.search(r"(\{\"name\"\s*:.*?\})", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return None
    return None


def run_tool_eval(model_path: str = "minival-v1"):
    print("\n" + "=" * 65)
    print(" MiniVal Function / Tool Calling Benchmark")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model = (model.half() if device == "cuda" else model.float()).eval().to(device)

    tools_desc = json.dumps(AVAILABLE_TOOLS, indent=2)
    system_prompt = (
        f"Kamu adalah MiniVal AI. Kamu memiliki akses ke tools berikut:\n{tools_desc}\n"
        "Jika pertanyaan memerlukan tool, panggil dengan format JSON:\n"
        "```json\n{\"name\": \"<nama_tool>\", \"arguments\": { ... }}\n```"
    )

    passed = 0
    for i, test in enumerate(TEST_CASES, 1):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test["prompt"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                inputs.input_ids,
                max_new_tokens=150,
                temperature=0.2,  # Low temperature untuk deterministik
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        tool_call = extract_tool_call(resp)

        is_correct = tool_call and tool_call.get("name") == test["expected_tool"]
        if is_correct:
            passed += 1

        status = " PASS" if is_correct else " FAIL"
        print(f"[{i}/{len(TEST_CASES)}] {status} | Prompt: {test['prompt']}")
        print(f"     Expected : {test['expected_tool']}")
        print(f"     Got      : {tool_call.get('name') if tool_call else 'No valid JSON tool call'}")
        print()

    print("=" * 65)
    print(f" Skor Akurasi Tool Calling: {passed}/{len(TEST_CASES)} ({passed/len(TEST_CASES)*100:.1f}%)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool Calling Evaluator")
    parser.add_argument("--model", default="minival-v1")
    args = parser.parse_args()
    run_tool_eval(args.model)
