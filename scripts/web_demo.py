"""
MiniVal Web Studio - Developer Edition
Antarmuka Chat & Riset LLM dengan estetika modern ala Linear / Cursor / Shadcn.
Palet warna: Soft Slate & Indigo (nyaman di mata developer, bebas kontras berlebihan).
"""
import html
import os
import sys
import time
from threading import Thread

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# Setup path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Force UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Page Configuration
st.set_page_config(
    page_title="MiniVal Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# SOFT DEVELOPER DARK THEME (LINEAR / CURSOR / SHADCN AESTHETICS)
# -----------------------------------------------------------------------------
DEV_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

/* Color Palette: Soft Slate (Theme Base) */
:root {
    --bg-base: #0f1117;
    --bg-surface: #161922;
    --bg-card: #1a1d28;
    --bg-card-hover: #212534;
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-focus: rgba(99, 102, 241, 0.4);
    --accent: #6366f1;
    --accent-soft: rgba(99, 102, 241, 0.12);
    --accent-light: #818cf8;
    --text-main: #e2e8f0;
    --text-muted: #94a3b8;
    --text-faint: #64748b;
    --radius-sm: 6px;
    --radius-md: 10px;
}

/* Base Typography & Background */
html, body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-main) !important;
}

/* Fix Material Symbols Icon Ligature (prevents raw text like 'keyb' / 'keyboard_arrow_left') */
[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="stIcon"] {
    font-family: 'Material Symbols Rounded' !important;
    font-weight: normal !important;
    font-style: normal !important;
    line-height: 1 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    display: inline-block !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    -webkit-font-feature-settings: 'liga' !important;
    -webkit-font-smoothing: antialiased !important;
}

/* Clean Header: Keep sidebar open/close buttons fully clickable */
header[data-testid="stHeader"] {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Hide Streamlit Clutter Only (Deploy button, hamburger menu, rainbow decoration) */
#MainMenu, footer, .stDeployButton, [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Sidebar OPEN / EXPAND Button (When sidebar is collapsed) */
[data-testid="stExpandSidebarButton"],
button[data-testid="stExpandSidebarButton"],
button[aria-label="Open sidebar"] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: fixed !important;
    top: 14px !important;
    left: 14px !important;
    z-index: 999999 !important;
    background-color: var(--bg-card) !important;
    color: var(--text-main) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: var(--radius-sm) !important;
    padding: 6px 10px !important;
    cursor: pointer !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.45) !important;
    transition: all 0.15s ease !important;
}

[data-testid="stExpandSidebarButton"]:hover,
button[data-testid="stExpandSidebarButton"]:hover,
button[aria-label="Open sidebar"]:hover {
    background-color: var(--bg-card-hover) !important;
    border-color: rgba(99, 102, 241, 0.5) !important;
    color: #ffffff !important;
    transform: scale(1.05);
}

/* Sidebar CLOSE / COLLAPSE Button (When sidebar is open) */
[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarCollapseButton"],
button[aria-label="Close sidebar"] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background-color: var(--bg-card) !important;
    color: var(--text-main) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: var(--radius-sm) !important;
    padding: 6px !important;
    cursor: pointer !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.15s ease !important;
}

[data-testid="stSidebarCollapseButton"]:hover,
button[data-testid="stSidebarCollapseButton"]:hover,
button[aria-label="Close sidebar"]:hover {
    background-color: var(--bg-card-hover) !important;
    border-color: rgba(99, 102, 241, 0.4) !important;
    color: #ffffff !important;
}

/* Main Container Layout */
.main .block-container {
    max-width: 900px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 7rem !important;
}

/* Sidebar - Sleek Soft Charcoal */
[data-testid="stSidebar"] {
    background-color: var(--bg-surface) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
[data-testid="stSidebarContent"] {
    padding-top: 1.25rem !important;
}
[data-testid="stSidebar"] hr {
    border-color: var(--border-subtle) !important;
    margin: 1.25rem 0 !important;
}

/* Modern Pill Badges */
.dev-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 9999px;
    padding: 3px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    line-height: 1.3;
    border: 1px solid var(--border-subtle);
    background: var(--bg-card);
    color: var(--text-muted);
}
.dev-badge-active {
    border-color: rgba(16, 185, 129, 0.3);
    background: rgba(16, 185, 129, 0.08);
    color: #34d399;
}
.dev-badge-indigo {
    border-color: rgba(99, 102, 241, 0.3);
    background: var(--accent-soft);
    color: var(--accent-light);
}

/* Header Section */
.dev-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding-bottom: 1.25rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--border-subtle);
}
.dev-header-left h1 {
    font-size: 1.5rem;
    font-weight: 600;
    letter-spacing: -0.025em;
    color: #f8fafc;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.dev-header-left p {
    font-size: 0.875rem;
    color: var(--text-muted);
    margin: 4px 0 0 0;
}

/* Starter Prompts Cards Grid */
.prompt-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    margin: 1.5rem 0 2rem 0;
}
.prompt-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 14px 16px;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}
.prompt-card:hover {
    border-color: var(--border-focus);
    background: var(--bg-card);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}
.prompt-card-header {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-main);
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.prompt-card-body {
    font-size: 0.8rem;
    color: var(--text-muted);
    line-height: 1.45;
}

/* Chat Message Containers */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 0.9rem 0 !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
}

/* User Message Card */
.msg-user {
    background: #1c202e;
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: var(--radius-md);
    padding: 12px 16px;
    color: #f1f5f9;
    font-size: 0.925rem;
    line-height: 1.6;
    display: inline-block;
    max-width: 100%;
}

/* Assistant Message Card */
.msg-assistant {
    color: #e2e8f0;
    font-size: 0.925rem;
    line-height: 1.65;
}

/* Reasoning (<think>) Collapsible Container */
.reasoning-box {
    margin: 6px 0 14px 0;
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: var(--radius-sm);
    background: rgba(15, 17, 23, 0.6);
    overflow: hidden;
}
.reasoning-box details {
    padding: 0;
}
.reasoning-box summary {
    padding: 8px 12px;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.775rem;
    font-weight: 500;
    color: var(--accent-light);
    display: flex;
    align-items: center;
    gap: 8px;
    user-select: none;
    transition: color 0.15s ease, background-color 0.15s ease;
}
.reasoning-box summary:hover {
    background: rgba(99, 102, 241, 0.08);
}
.reasoning-content {
    padding: 10px 14px;
    border-top: 1px solid rgba(99, 102, 241, 0.15);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    white-space: pre-wrap;
    line-height: 1.6;
    background: #0d0f14;
}

/* Telemetry & Speed Footer */
.telemetry-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-faint);
    margin-top: 8px;
}

/* Custom Chat Input Bar */
[data-testid="stChatInput"] {
    background-color: transparent !important;
}
[data-testid="stChatInput"] > div {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent), 0 4px 24px rgba(0, 0, 0, 0.45) !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text-main) !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-faint) !important;
}

/* Slider Controls: Soft Indigo Override */
[data-testid="stSlider"] > div > div > div > div {
    background-color: var(--accent) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background-color: #f8fafc !important;
    border: 2px solid var(--accent) !important;
    box-shadow: 0 0 8px rgba(99, 102, 241, 0.4) !important;
}

/* Selectbox Dropdowns */
[data-baseweb="select"] > div {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-main) !important;
}

/* Standard Buttons */
.stButton > button {
    background-color: var(--bg-card) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background-color: var(--bg-card-hover) !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
}

/* Hardware Telemetry Card in Sidebar */
.hw-box {
    margin-top: 1.5rem;
    padding: 12px;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    background: var(--bg-card);
}
.hw-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    color: var(--text-faint);
    letter-spacing: 0.05em;
}
.hw-value {
    font-size: 12.5px;
    font-weight: 500;
    color: var(--text-main);
    margin-top: 2px;
}
</style>
"""
st.markdown(DEV_THEME_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LOCALIZATIONS (ID / EN)
# -----------------------------------------------------------------------------
I18N = {
    'id': {
        'subtitle': 'Lingkungan Interaktif & Riset Model Bahasa Ringan',
        'sidebar_header': 'Parameter Inferensi',
        'model_label': 'Model Checkpoint',
        'temp_label': 'Temperature (Kreativitas)',
        'max_tokens': 'Maksimal Panjang Token',
        'thinking_label': 'Mode Penalaran (<think>)',
        'clear_btn': 'Bersihkan Riwayat',
        'input_placeholder': 'Ketik instruksi, prompt kode, atau pertanyaan untuk MiniVal...',
        'thinking_title': 'Rantai Penalaran (Reasoning Chain)',
        'starters': [
            ('⚡ Optimasi Kode Python', 'Bantu refactor fungsi pipeline data ini agar hemat memori & lebih cepat.'),
            ('🧠 Arsitektur RoPE & GQA', 'Jelaskan keuntungan matematis RoPE dibandingkan learned positional embeddings.'),
            ('📝 Evaluasi Fine-Tuning', 'Jelaskan perbedaan praktis antara SFT, LoRA rank 16, dan DPO alignment.'),
            ('🛡️ Keamanan Sistem LLM', 'Buat checklist validasi input untuk mencegah Prompt Injection di level API.')
        ]
    },
    'en': {
        'subtitle': 'High-Efficiency Sovereign Language Model Studio',
        'sidebar_header': 'Inference Parameters',
        'model_label': 'Model Checkpoint',
        'temp_label': 'Temperature (Creativity)',
        'max_tokens': 'Max Output Tokens',
        'thinking_label': 'Reasoning Mode (<think>)',
        'clear_btn': 'Clear Session',
        'input_placeholder': 'Enter instructions, code prompt, or questions for MiniVal...',
        'thinking_title': 'Reasoning Chain',
        'starters': [
            ('⚡ Python Code Optimization', 'Refactor this streaming dataset loader to minimize memory footprint.'),
            ('🧠 RoPE & GQA Architecture', 'Explain why RoPE with YaRN allows long context window extrapolation.'),
            ('📝 Fine-Tuning Evaluation', 'Compare practical tradeoffs between Full SFT, LoRA rank 16, and DPO.'),
            ('🛡️ LLM System Security', 'Provide production input sanitization rules against Prompt Injection attacks.')
        ]
    }
}

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; padding-bottom: 10px; margin-bottom: 4px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 18px;">⚡</span>
                <span style="font-weight: 700; font-size: 15px; color: #f8fafc; letter-spacing: -0.02em;">MiniVal</span>
                <span class="dev-badge dev-badge-indigo" style="font-size: 10px; padding: 1px 6px;">v1</span>
            </div>
            <span class="dev-badge dev-badge-active" style="font-size: 10px; padding: 1px 7px;">● Online</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Modern Segmented Control for Language
    lang_options = ["🇮🇩 Bahasa Indonesia", "🇬🇧 English"]
    selected_lang = st.segmented_control(
        "Language",
        lang_options,
        default=lang_options[0],
        label_visibility="collapsed"
    )
    lang_choice = selected_lang or lang_options[0]
    lang_key = 'id' if 'Indonesia' in lang_choice else 'en'
    T = I18N[lang_key]

    st.markdown("---")
    st.markdown(f"#### {T['sidebar_header']}")

    # Dynamic Model Discovery
    model_candidates = {}
    for search_path in [BASE_DIR, os.path.join(BASE_DIR, "out")]:
        if os.path.exists(search_path):
            for item in os.listdir(search_path):
                full_path = os.path.join(search_path, item)
                if os.path.isdir(full_path) and 'minival' in item.lower():
                    if any(f.endswith(('.safetensors', '.bin', '.pth')) for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))):
                        model_candidates[item] = full_path

    if not model_candidates:
        model_candidates = {"minival-v1 (Default)": os.path.join(BASE_DIR, "minival-v1")}

    selected_model_name = st.selectbox(T['model_label'], list(model_candidates.keys()), index=0)
    model_path = model_candidates[selected_model_name]

    # Hyperparameters
    temperature = st.slider(T['temp_label'], min_value=0.1, max_value=1.5, value=0.75, step=0.05)
    max_new_tokens = st.slider(T['max_tokens'], min_value=128, max_value=4096, value=1024, step=128)
    enable_thinking = st.checkbox(T['thinking_label'], value=False)

    st.markdown("---")
    if st.button(f"🗑️ {T['clear_btn']}", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Hardware Specs
    cuda_status = torch.cuda.is_available()
    dev_title = torch.cuda.get_device_name(0) if cuda_status else "CPU Only (Native Float32)"
    st.markdown(
        f"""
        <div class="hw-box">
            <div class="hw-label">Compute Device</div>
            <div class="hw-value">🎮 {dev_title}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# MODEL CACHE & INFERENCE ENGINE
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cached_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.half().eval().to(device)
    else:
        model = model.float().eval().to(device)
    return model, tokenizer, device

with st.spinner("Memuat bobot model..."):
    try:
        model, tokenizer, device = load_cached_model(model_path)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

# -----------------------------------------------------------------------------
# HERO HEADER
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="dev-header">
        <div class="dev-header-left">
            <h1>⚡ MiniVal Studio</h1>
            <p>{T['subtitle']}</p>
        </div>
        <div style="display: flex; gap: 8px; align-items: center;">
            <span class="dev-badge dev-badge-indigo">{selected_model_name}</span>
            <span class="dev-badge">FP16</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# STARTERS & MESSAGE HISTORY
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show starter cards when chat history is clean
if len(st.session_state.messages) == 0:
    st.markdown('<div class="prompt-grid">', unsafe_allow_html=True)
    cols = st.columns(2)
    for idx, (title, prompt_text) in enumerate(T['starters']):
        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="prompt-card">
                    <div class="prompt-card-header">{title}</div>
                    <div class="prompt-card-body">{prompt_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

# Helper function to parse <think> tag
def parse_reasoning_and_content(raw_text: str):
    think_content = ""
    clean_content = raw_text
    if "<think>" in raw_text:
        parts = raw_text.split("</think>")
        if len(parts) > 1:
            think_content = parts[0].replace("<think>", "").strip()
            clean_content = parts[1].strip()
        else:
            think_content = parts[0].replace("<think>", "").strip()
            clean_content = ""
    return think_content, clean_content


# Render existing messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            safe_user_content = html.escape(msg["content"]).replace("\n", "<br>")
            st.markdown(f'<div class="msg-user">{safe_user_content}</div>', unsafe_allow_html=True)
    else:
        with st.chat_message("assistant"):
            think_part, answer_part = parse_reasoning_and_content(msg["content"])
            if think_part:
                safe_think = html.escape(think_part)
                st.markdown(
                    f"""
                    <div class="reasoning-box">
                        <details>
                            <summary><span>🧠</span> <span>{T['thinking_title']}</span></summary>
                            <div class="reasoning-content">{safe_think}</div>
                        </details>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            if answer_part:
                safe_answer = html.escape(answer_part).replace("\n", "<br>")
                st.markdown(f'<div class="msg-assistant">{safe_answer}</div>', unsafe_allow_html=True)
            if "speed" in msg and msg["speed"] > 0:
                st.markdown(f'<div class="telemetry-tag">⚡ {msg["speed"]:.1f} tok/s · {msg.get("time", 0):.2f}s latency</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CHAT INPUT & GENERATION STREAM
# -----------------------------------------------------------------------------
if prompt := st.chat_input(T['input_placeholder']):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        safe_prompt = html.escape(prompt).replace("\n", "<br>")
        st.markdown(f'<div class="msg-user">{safe_prompt}</div>', unsafe_allow_html=True)

    # Prepare formatted input
    messages_payload = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    formatted_prompt = tokenizer.apply_chat_template(
        messages_payload,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=enable_thinking
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        input_ids=inputs.input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.90,
        do_sample=True,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
    thread.start()

    with st.chat_message("assistant"):
        live_box = st.empty()
        full_response = ""
        start_time = time.time()
        token_count = 0

        for token_text in streamer:
            full_response += token_text
            token_count += 1
            think_part, answer_part = parse_reasoning_and_content(full_response)

            # Live streaming feedback
            stream_html = ""
            if think_part:
                safe_think = html.escape(think_part)
                stream_html += f"""
                <div class="reasoning-box">
                    <details open>
                        <summary><span>🧠</span> <span>{T['thinking_title']}</span></summary>
                        <div class="reasoning-content">{safe_think}▌</div>
                    </details>
                </div>
                """
            if answer_part:
                safe_ans = html.escape(answer_part).replace("\n", "<br>")
                stream_html += f'<div class="msg-assistant">{safe_ans}▌</div>'
            elif not think_part:
                safe_full = html.escape(full_response).replace("\n", "<br>")
                stream_html += f'<div class="msg-assistant">{safe_full}▌</div>'

            live_box.markdown(stream_html, unsafe_allow_html=True)

        elapsed = time.time() - start_time
        speed = token_count / max(elapsed, 1e-4)

        # Final render after stream complete
        think_part, answer_part = parse_reasoning_and_content(full_response)
        final_html = ""
        if think_part:
            safe_think = html.escape(think_part)
            final_html += f"""
            <div class="reasoning-box">
                <details>
                    <summary><span>🧠</span> <span>{T['thinking_title']}</span></summary>
                    <div class="reasoning-content">{safe_think}</div>
                </details>
            </div>
            """
        display_text = answer_part if answer_part else full_response
        safe_display = html.escape(display_text).replace("\n", "<br>")
        final_html += f'<div class="msg-assistant">{safe_display}</div>'
        final_html += f'<div class="telemetry-tag">⚡ {speed:.1f} tok/s · {elapsed:.2f}s latency</div>'
        live_box.markdown(final_html, unsafe_allow_html=True)

    # Persist in state
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "speed": speed,
        "time": elapsed
    })
