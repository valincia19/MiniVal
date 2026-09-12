"""
Build Agent RL Dataset (Tool-Calling)
======================================
Generasi soal sintetis yang valid dijawab dengan tool call, atau tanpa tool.
Format (kompatibel skema minimind AgentRLDataset):
  {"messages": [{"role": "user", "content": ...}],
   "tools": [ {definisi JSONSchema} ],
   "gt": ["calculator_add"]        # nama tool yang diharapkan; [] = tanpa tool
  }
"""
import json
import random

random.seed(7)

TOOLS = [
    {"type": "function", "function": {"name": "calculator_add", "description": "Menjumlahkan dua angka",
     "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
     "required": ["a", "b"]}}},
    {"type": "function", "function": {"name": "calculator_mul", "description": "Mengalikan dua angka",
     "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
     "required": ["a", "b"]}}},
    {"type": "function", "function": {"name": "time_now", "description": "Menampilkan tanggal & waktu sekarang",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "random_dice", "description": "Melempar dadu dengan N sisi",
     "parameters": {"type": "object", "properties": {"sides": {"type": "number"}}, "required": ["sides"]}}},
]

NO_TOOL_QS = [
    "Apa itu ibu kota Indonesia?",
    "Saya sedang sedih, tolong semangatin dong.",
    "Jelaskan apa itu hujan.",
    "Buatkan kalimat sapaan untuk pagi hari.",
    "Siapa penemu lampu listrik?",
    "Apa itu internet?",
    "Beri tips biar sehat: singkat saja.",
]

def build_sample() -> dict:
    roll = random.random()
    if roll < 0.30:
        return {"messages": [{"role": "user", "content": random.choice(NO_TOOL_QS)}],
                "tools": TOOLS, "gt": []}
    if roll < 0.55:
        a, b = random.randint(2, 99), random.randint(2, 99)
        return {"messages": [{"role": "user", "content": f"Hitung {a} + {b} pakai calculator ya."}],
                "tools": TOOLS, "gt": ["calculator_add"]}
    if roll < 0.75:
        a, b = random.randint(2, 20), random.randint(2, 20)
        return {"messages": [{"role": "user", "content": f"{a} dikali {b} berapa? pakai calculator_mul"}],
                "tools": TOOLS, "gt": ["calculator_mul"]}
    if roll < 0.90:
        return {"messages": [{"role": "user", "content": "Jam berapa sekarang? pakai time_now"}],
                "tools": TOOLS, "gt": ["time_now"]}
    sides = random.choice([6, 12, 20])
    return {"messages": [{"role": "user", "content": f"Lempar dadu {sides} sisi sekali yuk"}],
            "tools": TOOLS, "gt": ["random_dice"]}

N = 600
with open("dataset/agent_rl.jsonl", "w", encoding="utf-8") as f:
    for _ in range(N):
        f.write(json.dumps(build_sample(), ensure_ascii=False) + "\n")

print(f"Menulis {N} sample agent RL ke dataset/agent_rl.jsonl")
