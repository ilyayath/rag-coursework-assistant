"""
ui/styles.py — усі CSS-стилі застосунку.
Викликати: inject_styles() один раз на початку app.py
"""

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

/* ── Reset & base ─────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: #08090d !important;
    color: #d4dae8 !important;
    font-family: 'Syne', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"]        { display: none; }
[data-testid="collapsedControl"] { display: none; }

::-webkit-scrollbar             { width: 4px; height: 4px; }
::-webkit-scrollbar-track       { background: transparent; }
::-webkit-scrollbar-thumb       { background: #1e2438; border-radius: 99px; }

/* ── Sidebar ───────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0b0d16 !important;
    border-right: 1px solid #161928 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 24px 18px 32px !important;
}

/* ── Main content area ─────────────────────────────────── */
[data-testid="stMainBlockContainer"] {
    padding: 28px 36px 110px !important;
    max-width: 860px !important;
    margin: 0 auto !important;
}

/* ── File uploader ─────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #0f1120 !important;
    border: 1px dashed #1e2438 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #2e4080 !important;
}
/* "Browse files" кнопка всередині аплоадера */
[data-testid="stFileUploader"] button {
    background: #141828 !important;
    border: 1px solid #1e2438 !important;
    color: #7a8aaa !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    border-radius: 6px !important;
    padding: 6px 14px !important;
}
[data-testid="stFileUploader"] button:hover {
    border-color: #4a7cf7 !important;
    color: #4a7cf7 !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] small {
    color: #353d54 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
}

/* ── Action buttons (Process / Clear) ──────────────────── */
[data-testid="stButton"] > button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 7px !important;
    padding: 9px 10px !important;
    width: 100% !important;
    transition: all 0.18s ease !important;
}

/* Primary — Обробити */
div[data-testid="stHorizontalBlock"] div:first-child [data-testid="stButton"] > button {
    background: #4a7cf7 !important;
    border: 1px solid #4a7cf7 !important;
    color: #fff !important;
    box-shadow: 0 2px 16px rgba(74,124,247,0.3) !important;
}
div[data-testid="stHorizontalBlock"] div:first-child [data-testid="stButton"] > button:hover {
    background: #5d8cf9 !important;
    box-shadow: 0 4px 24px rgba(74,124,247,0.45) !important;
    transform: translateY(-1px) !important;
}

/* Secondary — Очистити */
div[data-testid="stHorizontalBlock"] div:last-child [data-testid="stButton"] > button {
    background: transparent !important;
    border: 1px solid #1e2438 !important;
    color: #353d54 !important;
}
div[data-testid="stHorizontalBlock"] div:last-child [data-testid="stButton"] > button:hover {
    border-color: rgba(248,113,113,0.4) !important;
    color: #f87171 !important;
    background: rgba(248,113,113,0.05) !important;
}

/* ── Progress bar ──────────────────────────────────────── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #4a7cf7 0%, #7c5cf0 100%) !important;
    border-radius: 99px !important;
}

/* ── Chat messages ─────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 10px !important;
    margin-bottom: 8px !important;
    padding: 14px 18px !important;
    gap: 14px !important;
}

/* Override аватарів — прибираємо кольоровий фон */
[data-testid="stChatMessage"] [data-testid="stAvatar"] {
    background: #141828 !important;
    border: 1px solid #1e2438 !important;
    border-radius: 8px !important;
    width: 32px !important;
    height: 32px !important;
    font-size: 14px !important;
    color: #4a7cf7 !important;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #0f1120 !important;
    border: 1px solid #161928 !important;
}
/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: #0b0e1a !important;
    border: 1px solid rgba(74,124,247,0.12) !important;
    border-left: 2px solid rgba(74,124,247,0.35) !important;
}

/* ── Chat input ────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: #0b0d16 !important;
    border-top: 1px solid #161928 !important;
    padding: 14px 36px !important;
}
[data-testid="stChatInput"] textarea {
    background: #0f1120 !important;
    border: 1px solid #1e2438 !important;
    border-radius: 9px !important;
    color: #d4dae8 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 14px !important;
    caret-color: #4a7cf7 !important;
    resize: none !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(74,124,247,0.4) !important;
    box-shadow: 0 0 0 3px rgba(74,124,247,0.07) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #2a2f45 !important; }
[data-testid="stChatInput"] button {
    background: #4a7cf7 !important;
    border-radius: 8px !important;
    border: none !important;
}

/* ── Expander (sources) ────────────────────────────────── */
[data-testid="stExpander"] {
    background: #0a0c14 !important;
    border: 1px solid #161928 !important;
    border-radius: 7px !important;
    margin-top: 8px !important;
}
[data-testid="stExpander"] summary {
    color: #353d54 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stExpander"] summary:hover { color: #525b72 !important; }

/* ── Alerts ────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
}

/* ── Toast ─────────────────────────────────────────────── */
[data-testid="stToast"] {
    background: #141828 !important;
    border: 1px solid #1e2438 !important;
    border-radius: 8px !important;
    color: #d4dae8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
}

/* ── Spinner ───────────────────────────────────────────── */
[data-testid="stSpinner"] > div { border-top-color: #4a7cf7 !important; }
</style>
"""


def inject_styles() -> None:
    """Вставляє глобальні CSS-стилі у Streamlit-сторінку."""
    import streamlit as st
    st.markdown(STYLES, unsafe_allow_html=True)