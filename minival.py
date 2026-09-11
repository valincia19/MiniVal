"""
MiniVal Unified CLI / Antarmuka Terpadu MiniVal
Satu alat untuk semua: Chat, Web UI, Pelatihan Model (Pretrain/SFT/LoRA/DPO), OpenAI API Server, dan Diagnostik.
"""
import os
import sys

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import time
import argparse
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

from config import ModelPresets, TrainConfig
from model import MiniValConfig, MiniValForCausalLM
from dataset import UniversalDataset
from engine import MiniValTrainer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_info():
    """Tampilkan ringkasan status hardware dan lingkungan AI."""
    print("\n" + "=" * 60)
    print(" MiniVal System Diagnostics / Diagnostik MiniVal")
    print("=" * 60)
    print(f" Python         : {sys.version.split()[0]}")
    print(f" PyTorch        : {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f" CUDA Hardware  : {' Aktif' if cuda_ok else ' Tidak Aktif (CPU mode)'}")
    if cuda_ok:
        dev_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f" GPU Device     : {dev_name}")
        print(f" Total VRAM     : {vram:.2f} GB")
        print(f" bfloat16 Ready : {'Ya' if torch.cuda.is_bf16_supported() else 'Tidak (menggunakan float16)'}")
    print("=" * 60 + "\n")


def run_chat(model_path: str = "minival-v1", max_tokens: int = 1024, temp: float = 0.85):
    """Jalankan obrolan interaktif langsung di terminal."""
    full_path = os.path.join(BASE_DIR, model_path) if not os.path.isabs(model_path) else model_path
    if not os.path.exists(full_path):
        print(f" Folder model tidak ditemukan di: {full_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n Memuat model MiniVal dari '{model_path}' ke {device.upper()}...")
    tokenizer = AutoTokenizer.from_pretrained(full_path)
    model = AutoModelForCausalLM.from_pretrained(full_path, trust_remote_code=True)
    if device == "cuda":
        model = model.half().eval().to(device)
    else:
        model = model.float().eval().to(device)

    print("\n" + "=" * 60)
    print(" MiniVal Interactive Chat (Ketik 'exit' untuk keluar)")
    print("=" * 60 + "\n")

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    history = []

    while True:
        try:
            user_input = input("User : ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input or user_input.lower() in ['exit', 'quit', 'keluar']:
            print("\nSampai jumpa! ")
            break

        history.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        print("\nMiniVal : ", end="", flush=True)
        start_t = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_p=0.90,
                do_sample=True,
                streamer=streamer,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        elapsed = time.time() - start_t
        gen_tokens = len(outputs[0]) - len(inputs.input_ids[0])
        speed = gen_tokens / max(elapsed, 1e-4)
        print(f"\n[ Kecepatan: {speed:.1f} token/detik | {gen_tokens} token dalam {elapsed:.2f}s]\n")

        response_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        history.append({"role": "assistant", "content": response_text})


def run_web(port: int = 8501):
    """Luncurkan aplikasi WebUI Streamlit dengan penanganan shutdown instan."""
    web_script = os.path.join(BASE_DIR, "scripts", "web_demo.py")
    cmd = [sys.executable, "-m", "streamlit", "run", web_script, "--server.port", str(port)]
    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n Menghentikan WebUI MiniVal...")
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        print(" WebUI berhasil dihentikan.")


def run_serve(port: int = 8000, host: str = "127.0.0.1"):
    """Luncurkan OpenAI-compatible API server dengan penanganan shutdown instan."""
    serve_script = os.path.join(BASE_DIR, "scripts", "serve_api.py")
    cmd = [sys.executable, serve_script, "--host", host, "--port", str(port)]
    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n Menghentikan API Server MiniVal...")
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        print(" API Server berhasil dihentikan.")


def run_convert(args):
    """Konversi .pth ke HuggingFace safetensors format."""
    convert_script = os.path.join(BASE_DIR, "scripts", "convert_model.py")
    cmd = [sys.executable, convert_script, "--input", args.input, "--output", args.output, "--preset", args.preset]
    subprocess.run(cmd)


def _auto_train_config(data_path: str, args) -> dict:
    """
    Smart Auto-Config: ngitung hyperparameter terbaik secara otomatis
    berdasarkan ukuran dataset, device, dan RAM/VRAM yang tersedia.
    """
    import math

    # Hitung jumlah sampel data
    n_samples = 0
    try:
        with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    n_samples += 1
    except Exception:
        n_samples = 1000

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Deteksi VRAM/RAM
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gb >= 16:
            auto_batch, auto_seq, auto_preset = 16, 1024, "base"
        elif vram_gb >= 8:
            auto_batch, auto_seq, auto_preset = 8, 512, "base"
        elif vram_gb >= 4:
            auto_batch, auto_seq, auto_preset = 4, 512, "tiny"
        else:
            auto_batch, auto_seq, auto_preset = 2, 256, "tiny"
    else:
        # CPU: konservatif agar tidak freeze
        try:
            import psutil
            ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        except ImportError:
            ram_gb = 4.0
        if ram_gb >= 8:
            auto_batch, auto_seq, auto_preset = 4, 512, "tiny"
        else:
            auto_batch, auto_seq, auto_preset = 2, 256, "tiny"

    # Hitung epochs agar semua data ke-train minimal 1x
    # Target: ~1000-2000 optimizer steps total (cukup untuk SFT awal)
    steps_target = 1500
    steps_per_epoch = max(1, n_samples // auto_batch)
    auto_epochs = max(1, min(5, math.ceil(steps_target / steps_per_epoch)))

    # Override hanya jika user tidak set manual (masih di nilai default)
    out = {
        "batch_size": args.batch_size if args.batch_size != 16 else auto_batch,
        "max_seq_len": args.max_seq_len if args.max_seq_len != 1024 else auto_seq,
        "epochs": args.epochs if args.epochs != 2 else auto_epochs,
        "preset": args.preset if args.preset != "base" else auto_preset,
        "max_samples": args.max_samples,
        "n_samples": n_samples,
    }

    print(f"\n Auto-Config MiniVal (dataset: {n_samples:,} sampel | device: {device.upper()})")
    print(f"   preset={out['preset']} | batch={out['batch_size']} | seq_len={out['max_seq_len']} | epochs={out['epochs']}")
    print(f"   -> Estimasi steps: {n_samples // out['batch_size'] * out['epochs']:,}\n")
    return out


def run_train(args):
    # Hitung konfigurasi otomatis berdasarkan hardware & ukuran data
    cfg = _auto_train_config(args.data, args)
    preset_name = cfg["preset"]
    batch_size = cfg["batch_size"]
    max_seq_len = cfg["max_seq_len"]
    epochs = cfg["epochs"]

    model_cfg = ModelPresets.get(preset_name)
    train_cfg = TrainConfig(
        stage=args.stage,
        preset=preset_name,
        data_path=args.data,
        save_dir=args.save_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=args.lr,
        max_seq_len=max_seq_len,
    )

    if args.stage == "tokenizer":
        print(f" Melatih BPE Tokenizer dari korpus: {args.data}...")
        from tokenizers import ByteLevelBPETokenizer
        tok = ByteLevelBPETokenizer()
        tok.train(files=[args.data], vocab_size=6400, min_frequency=2,
                  special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
        save_path = os.path.join(BASE_DIR, "model")
        tok.save_model(save_path)
        print(f" Tokenizer berhasil disimpan di: {save_path}")
        return

    tokenizer_path = os.path.join(BASE_DIR, "model")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniValForCausalLM(model_cfg)

    # Muat bobot dasar jika ada
    if args.resume != "none" and os.path.exists(args.resume):
        weights = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=False)
        print(f" Memuat bobot dasar dari: {args.resume}")

    # Peringatan ramah CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and (args.max_samples is None or args.max_samples > 1000):
        print(" [Saran Mode CPU]: Training di CPU tanpa GPU akan lambat jika sampel terlalu banyak.")
        print("   Tips: Tambahkan '--max_samples 500 --preset tiny --batch_size 2 --max_seq_len 256'")
        print("   agar proses cepat selesai dan PC tetap adem serta tidak ngefreeze!\n")

    dataset = UniversalDataset(args.data, tokenizer, max_length=max_seq_len, max_samples=cfg["max_samples"])
    ref_model = None
    if args.stage == "dpo":
        ref_model = MiniValForCausalLM(model_cfg)
        if args.resume != "none" and os.path.exists(args.resume):
            ref_model.load_state_dict(weights, strict=False)

    trainer = MiniValTrainer(model, tokenizer, train_cfg, ref_model=ref_model)
    if args.stage == "dpo":
        trainer.fit_dpo(dataset)
    elif args.stage == "grpo":
        trainer.fit_grpo(dataset, num_generations=args.num_generations, max_new_tokens=args.max_new_tokens)
    elif args.stage == "distill":
        if not args.teacher:
            print(" Stage 'distill' memerlukan argumen --teacher <path/name teacher model>")
            return
        print(f" Memuat teacher model dari: {args.teacher}...")
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher, trust_remote_code=True)
        trainer.fit_distill(dataset, teacher)
    else:
        trainer.fit(dataset)


def main():
    parser = argparse.ArgumentParser(
        prog="minival",
        description="MiniVal: Arsitektur Pembuatan, Pelatihan, dan Evaluasi Model Bahasa Mandiri"
    )
    subparsers = parser.add_subparsers(dest="command", help="Perintah yang tersedia")

    # info
    subparsers.add_parser("info", help="Lihat status sistem, PyTorch & GPU")

    # chat
    p_chat = subparsers.add_parser("chat", help="Mulai dialog interaktif di terminal")
    p_chat.add_argument("--model", default="minival-v1", help="Folder path model (default: minival-v1)")
    p_chat.add_argument("--max_tokens", default=1024, type=int)
    p_chat.add_argument("--temperature", default=0.85, type=float)

    # web
    p_web = subparsers.add_parser("web", help="Buka antarmuka chat berbasis browser (Streamlit)")
    p_web.add_argument("--port", default=8501, type=int)

    # serve (OpenAI API)
    p_serve = subparsers.add_parser("serve", help="Jalankan OpenAI-compatible API server")
    p_serve.add_argument("--port", default=8000, type=int)
    p_serve.add_argument("--host", default="0.0.0.0", type=str)

    # convert
    p_conv = subparsers.add_parser("convert", help="Konversi bobot .pth ke HuggingFace safetensors")
    p_conv.add_argument("--input", required=True, help="File .pth input")
    p_conv.add_argument("--output", required=True, help="Folder tujuan model")
    p_conv.add_argument("--preset", default="base", choices=["tiny", "base", "pro", "moe"])

    # eval
    p_eval = subparsers.add_parser("eval", help="Jalankan benchmark throughput & akurasi kuantitatif")
    p_eval.add_argument("--model", default="minival-v1", help="Path model/checkpoint")
    p_eval.add_argument("--tokens", default=256, type=int, help="Max tokens per sample")

    # eval-tools
    p_tools = subparsers.add_parser("eval-tools", help="Benchmark kemampuan Tool / Function Calling")
    p_tools.add_argument("--model", default="minival-v1", help="Path model")

    # train
    p_train = subparsers.add_parser("train", help="Latih model (Pretrain, SFT, LoRA, DPO, GRPO, Distill, Tokenizer)")
    p_train.add_argument("stage", choices=["pretrain", "sft", "lora", "dpo", "grpo", "distill", "tokenizer"], help="Tahap pelatihan")
    p_train.add_argument("--data", required=True, help="Path ke file dataset (.txt atau .jsonl)")
    p_train.add_argument("--preset", default="base", choices=["tiny", "base", "pro", "moe"], help="Ukuran model")
    p_train.add_argument("--epochs", default=2, type=int)
    p_train.add_argument("--batch_size", default=16, type=int)
    p_train.add_argument("--lr", default=2e-5, type=float)
    p_train.add_argument("--max_seq_len", default=1024, type=int)
    p_train.add_argument("--save_dir", default="out", type=str)
    p_train.add_argument("--resume", default="none", type=str)
    p_train.add_argument("--max_samples", default=None, type=int, help="Batasi jumlah sampel data (misal 500 untuk hemat CPU/RAM)")
    p_train.add_argument("--teacher", default=None, type=str, help="Path teacher model (khusus stage=distill)")
    p_train.add_argument("--num_generations", default=4, type=int, help="Rollouts per prompt (khusus stage=grpo)")
    p_train.add_argument("--max_new_tokens", default=256, type=int, help="Panjang completion max per rollout (khusus grpo)")

    args = parser.parse_args()

    if args.command == "info":
        run_info()
    elif args.command == "chat":
        run_chat(args.model, args.max_tokens, args.temperature)
    elif args.command == "web":
        run_web(args.port)
    elif args.command == "serve":
        run_serve(args.port, args.host)
    elif args.command == "convert":
        run_convert(args)
    elif args.command == "eval":
        eval_script = os.path.join(BASE_DIR, "scripts", "eval_val.py")
        subprocess.run([sys.executable, eval_script, "--model", args.model, "--tokens", str(args.tokens)])
    elif args.command == "eval-tools":
        tool_script = os.path.join(BASE_DIR, "scripts", "eval_toolcall.py")
        subprocess.run([sys.executable, tool_script, "--model", args.model])
    elif args.command == "train":
        run_train(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
