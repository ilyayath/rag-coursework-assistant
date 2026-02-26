"""
ui/components.py — HTML-компоненти для відображення через st.markdown.
Всі функції повертають рядок HTML або одразу рендерять через st.markdown.
"""
import streamlit as st


# ── Заголовок сторінки ────────────────────────────────────────────────────────

def render_page_header() -> None:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
        <span style="
            font-family:'IBM Plex Mono',monospace;
            font-size:11px; font-weight:500;
            letter-spacing:0.14em; text-transform:uppercase;
            color:#4a7cf7;
            background:rgba(74,124,247,0.08);
            border:1px solid rgba(74,124,247,0.18);
            padding:4px 11px; border-radius:4px;">
            RAG · v1.0
        </span>
        <h1 style="
            margin:0; font-size:20px; font-weight:800;
            color:#e0e6f5; letter-spacing:-0.02em;
            font-family:'Syne',sans-serif;">
            Асистент з документів
        </h1>
    </div>
    <p style="
        margin:0 0 18px; font-size:13px; color:#353d54;
        font-family:'IBM Plex Mono',monospace; letter-spacing:0.01em;">
        Відповіді беруться виключно з ваших файлів
    </p>
    <div style="height:1px; background:linear-gradient(90deg,#1a1f32,transparent);
                margin-bottom:22px;"></div>
    """, unsafe_allow_html=True)


# ── Sidebar: логотип ──────────────────────────────────────────────────────────

def render_sidebar_logo() -> None:
    st.markdown("""
    <div style="
        display:flex; align-items:center; gap:8px;
        margin-bottom:20px; padding-bottom:16px;
        border-bottom:1px solid #161928;">
        <span style="
            font-size:16px; font-weight:800; color:#e0e6f5;
            font-family:'Syne',sans-serif; letter-spacing:-0.01em;">
            ◈ DocMind
        </span>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar: мітки секцій ─────────────────────────────────────────────────────

def section_label(text: str, margin_top: int = 18) -> None:
    st.markdown(f"""
    <div style="
        font-family:'IBM Plex Mono',monospace;
        font-size:9px; font-weight:500;
        letter-spacing:0.16em; text-transform:uppercase;
        color:#252a3e; margin:{margin_top}px 0 8px;">
        {text}
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar: статистика ───────────────────────────────────────────────────────

def render_stats(doc_count: int, chunk_count: int) -> None:
    st.markdown(f"""
    <div style="display:flex; gap:8px; margin-top:4px;">
        <div style="
            flex:1; background:#0f1120;
            border:1px solid #161928; border-radius:8px;
            padding:11px 14px;">
            <div style="
                font-family:'IBM Plex Mono',monospace;
                font-size:22px; font-weight:500; color:#4a7cf7;
                line-height:1;">
                {doc_count}
            </div>
            <div style="
                font-size:9px; color:#252a3e; margin-top:5px;
                letter-spacing:0.12em; text-transform:uppercase;
                font-family:'IBM Plex Mono',monospace;">
                Документів
            </div>
        </div>
        <div style="
            flex:1; background:#0f1120;
            border:1px solid #161928; border-radius:8px;
            padding:11px 14px;">
            <div style="
                font-family:'IBM Plex Mono',monospace;
                font-size:22px; font-weight:500; color:#4a7cf7;
                line-height:1;">
                {chunk_count}
            </div>
            <div style="
                font-size:9px; color:#252a3e; margin-top:5px;
                letter-spacing:0.12em; text-transform:uppercase;
                font-family:'IBM Plex Mono',monospace;">
                Фрагментів
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar: список завантажених файлів ───────────────────────────────────────

def render_doc_list(doc_names: list[str]) -> None:
    if not doc_names:
        return
    badges = "".join(
        f"""<div style="
            display:flex; align-items:center; gap:6px;
            background:rgba(74,124,247,0.06);
            border:1px solid rgba(74,124,247,0.14);
            border-radius:5px; padding:5px 10px;
            margin-bottom:4px;
            font-family:'IBM Plex Mono',monospace;
            font-size:11px; color:#4a7cf7;
            overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
            <span style="opacity:0.5;">📄</span>{name}
        </div>"""
        for name in doc_names
    )
    st.markdown(badges, unsafe_allow_html=True)


# ── Порожній стан чату ────────────────────────────────────────────────────────

def render_empty_state() -> None:
    st.markdown("""
    <div style="
        display:flex; flex-direction:column;
        align-items:center; justify-content:center;
        padding:90px 20px; text-align:center;
        color:#1e2438;">
        <div style="
            width:52px; height:52px;
            border:1px solid #161928; border-radius:12px;
            display:flex; align-items:center; justify-content:center;
            font-size:22px; margin-bottom:16px;
            background:#0b0d16;">
            ◈
        </div>
        <div style="
            font-size:14px; font-weight:700; color:#1e2438;
            letter-spacing:-0.01em; margin-bottom:8px;">
            База документів порожня
        </div>
        <div style="
            font-family:'IBM Plex Mono',monospace;
            font-size:11px; color:#161928; line-height:1.7;">
            ← Завантажте PDF або TXT на бічній панелі<br>
            і натисніть «Обробити»
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Джерела у повідомленні ────────────────────────────────────────────────────

def render_sources(sources: list[dict]) -> None:
    """Відображає список джерел всередині expander."""
    if not sources:
        return
    with st.expander(f"↗  джерела · {len(sources)} фрагм."):
        for s in sources:
            st.markdown(
                f"""<div style="
                    font-family:'IBM Plex Mono',monospace;
                    font-size:11px; color:#4a7cf7;
                    background:rgba(74,124,247,0.05);
                    border:1px solid rgba(74,124,247,0.1);
                    border-radius:5px; padding:5px 10px;
                    margin-bottom:4px;">
                    {s.get('source','?')}
                    <span style="color:#252a3e; margin-left:8px;">
                        стор. {s.get('page','?')}
                    </span>
                </div>""",
                unsafe_allow_html=True,
            )