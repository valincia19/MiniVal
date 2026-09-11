"""
MiniVal GGUF Exporter & Ollama Helper
======================================
Alat untuk mengonversi checkpoint MiniVal ke format GGUF (llama.cpp)
dan membuat Modelfile otomatis untuk dijalankan langsung di Ollama!

Penggunaan:
    python scripts/export_gguf.py --model minival-v1 --quant q4_k_m
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def generate_ollama_modelfile(gguf_filename: str, out_dir: str):
    """Buat file Modelfile untuk Ollama secara otomatis."""
    modelfile_path = os.path.join(out_dir, "Modelfile")
    content = f"""FROM ./{gguf_filename}

# Set template percakapan MiniVal
TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ if .Thinking }}}}<think>
{{{{ end }}}}{{{{ .Response }}}}<|im_end|>
\"\"\"

# Parameter inferensi default
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f" Modelfile Ollama dibuat di: {modelfile_path}")
    print(f" Jalankan di Ollama dengan:")
    print(f"   cd {out_dir}")
    print(f"   ollama create minival -f Modelfile")
    print(f"   ollama run minival")


def export_to_gguf(model_dir: str, output_path: str, quant_type: str = "q4_k_m"):
    print("\n" + "=" * 65)
    print(" MiniVal GGUF Exporter (Ollama / LM Studio / llama.cpp)")
    print("=" * 65)

    if not os.path.exists(model_dir):
        print(f" Folder model tidak ditemukan: {model_dir}")
        return

    out_folder = os.path.dirname(output_path) or "."
    os.makedirs(out_folder, exist_ok=True)

    # Cek ketersediaan llama.cpp convert script
    convert_script = None
    possible_paths = [
        os.environ.get("LLAMA_CPP_CONVERT_SCRIPT", ""),
        "convert_hf_to_gguf.py",
        os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
        shutil.which("convert_hf_to_gguf.py") or "",
    ]
    for p in possible_paths:
        if p and os.path.exists(p):
            convert_script = p
            break

    if not convert_script:
        print("\n Script 'convert_hf_to_gguf.py' dari llama.cpp belum ditemukan.")
        print("   Cara setup mudah:")
        print("   1. git clone https://github.com/ggerganov/llama.cpp")
        print("   2. pip install -r llama.cpp/requirements.txt")
        print("   3. python llama.cpp/convert_hf_to_gguf.py " + model_dir + f" --outfile {output_path} --outtype {quant_type}")
        print("\n   Sebagai alternatif, Modelfile untuk Ollama tetap kami siapkan di bawah:")
        gguf_name = os.path.basename(output_path)
        generate_ollama_modelfile(gguf_name, out_folder)
        return

    cmd = [
        sys.executable,
        convert_script,
        model_dir,
        "--outfile", output_path,
        "--outtype", quant_type,
    ]
    print(f" Menjalankan konversi: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n Berhasil diexport ke GGUF: {output_path}")
        generate_ollama_modelfile(os.path.basename(output_path), out_folder)
    else:
        print("\n Gagal saat menjalankan konversi llama.cpp.")


def main():
    parser = argparse.ArgumentParser(description="Export MiniVal model ke format GGUF")
    parser.add_argument("--model", default="minival-v1", help="Folder model HuggingFace")
    parser.add_argument("--output", default="out/minival-v1.gguf", help="Path file GGUF tujuan")
    parser.add_argument("--quant", default="f16", choices=["f16", "f32", "q8_0", "q4_k_m"], help="Tipe quantization")
    args = parser.parse_args()

    export_to_gguf(args.model, args.output, args.quant)


if __name__ == "__main__":
    main()
