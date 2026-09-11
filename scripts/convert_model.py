"""
MiniVal Model Converter
Konversi bobot PyTorch (*.pth) ke format HuggingFace Transformers (safetensors).
Memungkinkan model hasil training kamu dipakai di Ollama, vLLM, llama.cpp, atau HuggingFace Hub.
"""
import os
import sys
import argparse
import torch
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import ModelPresets
from model import MiniValConfig, MiniValForCausalLM


def convert_pth_to_hf(pth_path: str, output_dir: str, preset: str = "base"):
    if not os.path.exists(pth_path):
        print(f" File {pth_path} tidak ditemukan!")
        return

    os.makedirs(output_dir, exist_ok=True)
    config = ModelPresets.get(preset)

    print(f" Mengonversi {pth_path} ke format HuggingFace...")
    model = MiniValForCausalLM(config)
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    # Simpan bobot HuggingFace
    model.half().save_pretrained(output_dir, safe_serialization=True)

    # Salin tokenizer
    tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(output_dir)

    # Salin file arsitektur model_minival.py agar HuggingFace auto_map langsung jalan
    import shutil
    arch_src = os.path.join(tokenizer_path, "model_minival.py")
    if os.path.exists(arch_src):
        shutil.copy(arch_src, os.path.join(output_dir, "model_minival.py"))

    # Tambahkan auto_map ke config.json
    cfg_json_path = os.path.join(output_dir, "config.json")
    if os.path.exists(cfg_json_path):
        import json
        with open(cfg_json_path, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        cfg_data["auto_map"] = {
            "AutoConfig": "model_minival.MiniValConfig",
            "AutoModelForCausalLM": "model_minival.MiniValForCausalLM"
        }
        with open(cfg_json_path, "w", encoding="utf-8") as f:
            json.dump(cfg_data, f, indent=2)

    print(f" Berhasil dikonversi ke: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="MiniVal Checkpoint Converter")
    parser.add_argument("--input", required=True, help="Path ke file .pth")
    parser.add_argument("--output", required=True, help="Folder tujuan untuk model HuggingFace")
    parser.add_argument("--preset", default="base", choices=["tiny", "base", "pro", "moe"])
    args = parser.parse_args()

    convert_pth_to_hf(args.input, args.output, args.preset)


if __name__ == '__main__':
    main()
