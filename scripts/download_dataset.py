"""
MiniVal Dataset Downloader & Preparer
======================================
Download dan konversi dataset open-source populer langsung ke format MiniVal:
  - alpaca-id : Instruksi percakapan Bahasa Indonesia
  - gsm8k     : Soal penalaran matematika tingkat SD (GSM8K)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))


def download_alpaca_id():
    """Download Alpaca Cleaned Bahasa Indonesia dengan deteksi skema kunci otomatis."""
    out_path = os.path.join(DATASET_DIR, "alpaca_id.jsonl")

    try:
        from datasets import load_dataset
        print(" Memuat dataset 'FreedomIntelligence/alpaca-gpt4-indonesian' dari cache/HuggingFace...")
        ds = load_dataset("FreedomIntelligence/alpaca-gpt4-indonesian", split="train")
    except Exception as e:
        print(f" Gagal memuat via datasets: {e}")
        return

    if len(ds) == 0:
        print(" Dataset kosong!")
        return

    first_sample = ds[0]
    print(f" Struktur field terdeteksi: {list(first_sample.keys())}")

    def get_field(item, candidates):
        for k in candidates:
            val = item.get(k)
            if val and str(val).strip():
                return str(val).strip()
        return ""

    inst_keys = ["instruction_id", "instruction", "prompt", "question", "query"]
    inp_keys  = ["input_id", "input", "context"]
    out_keys  = ["output_id", "output", "response", "answer", "completion"]

    print(f" Mengonversi {len(ds)} sampel ke format MiniVal universal...")
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in ds:
            # Jika sudah berbentuk list percakapan
            if "conversations" in item and isinstance(item["conversations"], list):
                f.write(json.dumps({"conversations": item["conversations"]}, ensure_ascii=False) + "\n")
                count += 1
                continue

            instruction = get_field(item, inst_keys)
            user_input  = get_field(item, inp_keys)
            output      = get_field(item, out_keys)

            if not instruction or not output:
                # Fallback: jika hanya ada 2 field teks apapun
                text_vals = [str(v).strip() for v in item.values() if isinstance(v, str) and len(str(v).strip()) > 3]
                if len(text_vals) >= 2:
                    instruction = text_vals[0]
                    output = text_vals[1]
                else:
                    continue

            full_user = f"{instruction}\n\nInput: {user_input}" if user_input else instruction
            entry = {
                "conversations": [
                    {"role": "user", "content": full_user},
                    {"role": "assistant", "content": output},
                ]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    print(f" Selesai! {count} sampel tersimpan di: {out_path}")
    print(f" Latih model dengan:")
    print(f"   python minival.py train sft --data dataset/alpaca_id.jsonl --preset base")


def download_gsm8k():
    """Download GSM8K via HuggingFace datasets."""
    out_path = os.path.join(DATASET_DIR, "gsm8k_rl.jsonl")
    try:
        from datasets import load_dataset
    except ImportError:
        print(" Library 'datasets' belum terinstall. Jalankan: pip install datasets")
        return

    print(" Mengunduh GSM8K (grade school math)...")
    ds = load_dataset("openai/gsm8k", "main", split="train")

    print(f" Mengonversi {len(ds)} sampel ke format RL MiniVal...")
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in ds:
            question = item.get("question", "").strip()
            if not question:
                continue
            entry = {"prompt": question}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    print(f" Selesai! {count} prompt tersimpan di: {out_path}")
    print(f" Latih reasoning model dengan:")
    print(f"   python minival.py train grpo --data dataset/gsm8k_rl.jsonl --num_generations 4")


def main():
    parser = argparse.ArgumentParser(description="MiniVal Dataset Downloader")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["alpaca-id", "gsm8k"],
        help="Nama dataset yang ingin diunduh",
    )
    args = parser.parse_args()

    os.makedirs(DATASET_DIR, exist_ok=True)
    if args.dataset == "alpaca-id":
        download_alpaca_id()
    elif args.dataset == "gsm8k":
        download_gsm8k()


if __name__ == "__main__":
    main()
