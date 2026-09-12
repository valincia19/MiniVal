"""
Build DPO dataset dari alpaca_id.jsonl
=======================================
chosen   = jawaban asli (peran 'gpt')
rejected = korupsi jawaban: lazy-truncate / vague / sloppy mengacak
Output   : dataset/dpo.jsonl (format {"chosen": [...], "rejected": [...]})
"""
import json
import random
import os
import re

random.seed(42)

SRC = "dataset/alpaca_id.jsonl"
OUT = "dataset/dpo.jsonl"
TARGET = 2000

VAGUE_REPLIES = [
    "Saya kurang yakin jawabannya, coba tanyakan hal lain.",
    "Itu pertanyaan yang rumit, saya tidak bisa menjelaskannya sekarang.",
    "Hmm, mungkin ada, mungkin tidak. Saya kurang paham soal ini.",
    "Jawabannya kurang lebih seperti itu saja.",
    "Langsung coba sendiri di internet ya, saya tidak tahu detailnya.",
]

def make_rejected(text: str, kind: str) -> str:
    if kind == "vague":
        return random.choice(VAGUE_REPLIES)
    if kind == "lazy":
        # potong 1-2 kalimat pertama, tambahkan "dan seterusnya"
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        keep = " ".join(parts[:random.randint(1, 2)])
        return keep + (" dan seterusnya." if not keep.endswith(".") else "")
    # sloppy: hilangkan struktur, code block, kapital, tanda baca
    t = re.sub(r"```.*?```", "kode-nya bentar", text, flags=re.DOTALL)
    t = re.sub(r"\*\*?|#+|^[-*]\s", "", t, flags=re.MULTILINE)
    t = t.lower().replace(".", "").replace(",", "").replace("\n", " ")
    return t.strip()

pairs = []
with open(SRC, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        conv = rec.get("conversations") or []
        msgs = [{"role": ("user" if m.get("from") in (None, "human") else "assistant"),
                 "content": str(m.get("value") or m.get("content") or "").strip()}
                for m in conv if (m.get("value") or m.get("content"))]
        if not msgs:
            continue
        prompt = [msgs[0]]
        chosen = msgs[-1]
        if prompt[0]["role"] != "user" or chosen["role"] != "assistant":
            continue
        ans = chosen["content"]
        if len(ans) < 120:            # jawaban derita kecil, korupsi-nya terlalu hampir
            continue
        kind = random.choice(["lazy", "vague", "sloppy", "lazy", "sloppy"])
        rej = make_rejected(ans, kind)
        if rej.strip() == ans.strip():
            continue
        pairs.append({"chosen": prompt + [chosen], "rejected": prompt + [{"role": "assistant", "content": rej}]})

random.shuffle(pairs)
pairs = pairs[:TARGET]

with open(OUT, "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"Menulis {len(pairs)} pasangan DPO ke {OUT}")
