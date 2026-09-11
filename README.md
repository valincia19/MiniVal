# MiniVal ⚡

Proyek iseng belajar bikin LLM dari nol.

> Bukan untuk produksi. Bukan state-of-the-art. Murni buat ngerti cara kerja LLM dari dalamnya.

---

## Filosofi

- **Satu CLI** — semua via `python minival.py <command>`, nggak perlu hafal folder
- **Satu engine** — `engine.py` handle semua stage training
- **Modular** — tiap komponen (`rollout.py`, `rewards.py`) bisa dipakai sendiri
- **Bersih** — kode dan komentar bahasa Indonesia & Inggris

---

## Struktur

```
MiniVal/
├── minival.py          # Entry point CLI
├── config.py           # Presets model & training config
├── engine.py           # Trainer (Pretrain, SFT, LoRA, DPO, GRPO, Distill)
├── rollout.py          # Rollout engine untuk RL
├── rewards.py          # Reward functions modular
├── model/
│   ├── model_minival.py  # Arsitektur transformer
│   └── model_lora.py     # LoRA adapter
├── dataset/
│   └── universal.py    # Dataset loader universal (.txt / .jsonl)
└── scripts/
    ├── web_demo.py       # Streamlit chat UI
    ├── serve_api.py      # OpenAI-compatible API server
    ├── convert_model.py  # Konversi .pth → safetensors
    ├── eval_val.py       # Benchmark throughput
    └── eval_toolcall.py  # Benchmark function calling
```

---

## Quickstart

```powershell
cd D:\Valincia\Valinc-LLM-Lab\MiniVal
.\.venv\Scripts\Activate.ps1
```

```powershell
# Lihat info GPU
python minival.py info

# Chat di terminal
python minival.py chat

# Web UI (browser)
python minival.py web
```

---

## Training

```powershell
# Pretrain dari teks mentah
python minival.py train pretrain --data dataset/sample_data.jsonl --preset tiny --epochs 1

# SFT
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

---

## Evaluasi

```powershell
# Benchmark throughput & output quality
python minival.py eval --model minival-v1

# Test kemampuan function calling
python minival.py eval-tools --model minival-v1
```

---

## API Server (OpenAI-compatible)

```powershell
python minival.py serve --port 8000
```

Konek dari Open-WebUI, Cursor, atau app lain ke `http://localhost:8000/v1`.

---

## Model Presets

| Preset | Parameter | Cocok untuk |
|--------|-----------|-------------|
| `tiny` | ~26M | Eksperimen cepat, laptop tanpa GPU |
| `base` | ~108M | Training serius, GPU 6GB+ |
| `moe`  | ~108M aktif | Coba arsitektur MoE |

---

## Dependencies

```
torch, transformers, fastapi, uvicorn, streamlit, pydantic>=2
```

Install: `pip install -r requirements.txt`

---

*Dibuat iseng. Kalau ada yang mau benerin atau nambahin, silakan.*
