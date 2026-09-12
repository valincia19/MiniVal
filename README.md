# MiniVal ⚡

[![CI](https://github.com/valincia19/MiniVal/actions/workflows/ci.yml/badge.svg)](https://github.com/valincia19/MiniVal/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)

Proyek iseng belajar bikin LLM dari nol.

> Bukan untuk produksi. Bukan state-of-the-art. Murni buat ngerti cara kerja LLM dari dalamnya.
> *An educational, from-scratch LLM implementation in PyTorch (Pretrain, SFT, LoRA, DPO, GRPO reasoning RL, MoE, and OpenAI-compatible API).*

---

## Filosofi

- **Satu CLI** — semua via `python minival.py <command>`, nggak perlu hafal folder
- **Satu engine** — `engine.py` handle semua stage training
- **Modular** — tiap komponen (`rollout.py`, `rewards.py`) bisa dipakai sendiri
- **Bersih** — kode dan komentar bahasa Indonesia & Inggris

---

## Struktur

```text
MiniVal/
├── minival.py          # Entry point CLI
├── config.py           # Presets model & training config
├── engine.py           # Trainer (Pretrain, SFT, LoRA, DPO, GRPO, Distill)
├── rollout.py          # Rollout engine untuk RL
├── rewards.py          # Reward functions modular
├── model/
│   ├── model_minival.py  # Arsitektur transformer (GQA, RoPE/YaRN, MoE)
│   └── model_lora.py     # LoRA adapter
├── dataset/
│   └── universal.py    # Dataset loader universal (.txt / .jsonl)
├── scripts/
│   ├── web_demo.py       # Streamlit chat UI
│   ├── serve_api.py      # OpenAI-compatible API server
│   ├── convert_model.py  # Konversi .pth → safetensors
│   ├── eval_val.py       # Benchmark throughput
│   └── eval_toolcall.py  # Benchmark function calling
└── test_verification.py # 7-suite automated regression test
```

---

## Quickstart

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/valincia19/MiniVal.git
cd MiniVal

# Buat virtual environment
python -m venv .venv

# Aktivasi virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Jalankan MiniVal

```bash
# Lihat info hardware & GPU
python minival.py info

# Chat interaktif di terminal
python minival.py chat

# Web UI (browser via Streamlit)
python minival.py web
```

---

## Menjalankan Pengujian (Testing)

MiniVal dilengkapi dengan test suite regresi lengkap yang mencakup arsitektur, MoE aux loss, LoRA, rewards engine, token masking, rollout logps, dan skema OpenAI API:

```bash
python test_verification.py
```

Jika menggunakan `pytest`:
```bash
pytest test_verification.py
```

---

## Training

```bash
# Pretrain dari teks mentah
python minival.py train pretrain --data dataset/sample_data.jsonl --preset tiny --epochs 1

# SFT (Instruction Tuning)
python minival.py train sft --data dataset/sample_data.jsonl --preset base --epochs 2

# LoRA (hemat VRAM)
python minival.py train lora --data dataset/sample_data.jsonl --preset base

# DPO (preference tuning)
python minival.py train dpo --data dataset/dpo.jsonl

# GRPO (reasoning RL, ala DeepSeek-R1)
python minival.py train grpo --data dataset/sample_data.jsonl --num_generations 4

# Knowledge Distillation dari model besar
python minival.py train distill --data dataset/sample_data.jsonl --teacher Qwen/Qwen2-1.5B
```

> [!TIP]
> **Mode CPU**: Tambahkan `--max_samples 500 --preset tiny --batch_size 2 --max_seq_len 256` agar proses training di laptop tanpa GPU tetap ringan dan cepat selesai.

---

## Evaluasi

```bash
# Benchmark throughput & output quality
python minival.py eval --model minival-v1

# Test kemampuan function calling
python minival.py eval-tools --model minival-v1
```

---

## API Server (OpenAI-compatible)

```bash
python minival.py serve --port 8000
```

Konek dari Open-WebUI, Cursor, Continue.dev, atau app lain ke `http://localhost:8000/v1`.

Untuk mengaktifkan autentikasi Bearer token:
```bash
export MINIVAL_API_KEY="your-secret-key"   # Linux/macOS
$env:MINIVAL_API_KEY="your-secret-key"     # Windows PowerShell
```

---

## Model Presets

| Preset | Parameter | Cocok untuk |
|--------|-----------|-------------|
| `tiny` | ~26M | Eksperimen cepat, laptop tanpa GPU |
| `base` | ~108M | Training serius, GPU 6GB+ |
| `pro`  | ~280M | Model lebih besar, multi-GPU / VRAM 12GB+ |
| `moe`  | ~108M aktif | Coba arsitektur Mixture-of-Experts |

---

## Kontribusi & Komunitas

Kontribusi selalu disambut hangat! Silakan baca panduan berikut sebelum berkontribusi:

- [Panduan Kontribusi (CONTRIBUTING.md)](CONTRIBUTING.md) — Alur kerja git, standar commit, pembuatan PR, dan pelaporan isu.
- [Code of Conduct](CODE_OF_CONDUCT.md) — Standar komunitas dan etika kolaborasi.
- [Security Policy](SECURITY.md) — Prosedur pelaporan celah keamanan secara privat dan bertanggung jawab.

---

## Lisensi (License)

Hak cipta (c) 2026 valincia19 dan kontributor MiniVal.

*(Catatan Maintainer: Lisensi formal open source seperti **Apache 2.0** atau **MIT** dapat dipilih dan ditentukan oleh pemilik repositori pada berkas `LICENSE`).*

---

*Dibuat iseng. Kalau ada yang mau benerin atau nambahin, silakan.*
