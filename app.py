"""
app.py — точка входу застосунку.
Тут лише page_config, ін'єкція стилів та виклик UI-модулів.
"""

# ── Page config — ОБОВ'ЯЗКОВО першим ─────────────────────────────────────────
import streamlit as st

st.set_page_config(
    page_title="DocMind · RAG Assistant",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Стилі ─────────────────────────────────────────────────────────────────────
from ui.styles import inject_styles
inject_styles()

# ── Імпорт логіки ─────────────────────────────────────────────────────────────
try:
    from src.vector_store import VectorStore
    from src.rag_chain    import RAGChain
    from src.logger       import get_logger
except ImportError as e:
    st.error(f"Помилка імпорту модулів src: {e}")
    st.stop()

# ── Імпорт UI-модулів ─────────────────────────────────────────────────────────
from ui.components import render_page_header
from ui.sidebar    import render_sidebar
from ui.chat       import render_chat

logger = get_logger("App")


# ── Ініціалізація стану сесії ─────────────────────────────────────────────────
def _init_session() -> None:
    defaults = {
        "messages":     [],
        "doc_names":    [],
        "total_chunks": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ── Ініціалізація системних компонентів (кешується) ──────────────────────────
@st.cache_resource(show_spinner=False)
def _load_system():
    """
    Завантажує VectorStore та RAGChain один раз за сесію.
    Результат кешується — при рефреші не перезавантажується.
    """
    return VectorStore(), RAGChain()


# ── Головна функція ───────────────────────────────────────────────────────────
def main() -> None:
    _init_session()

    # Завантаження моделей
    with st.spinner("Завантаження моделей…"):
        try:
            vector_store, rag_chain = _load_system()
        except Exception as e:
            st.error(f"Критична помилка запуску: {e}")
            st.info("Перевірте, чи запущено Ollama і встановлені всі залежності.")
            st.stop()

    # Рендер сторінки
    render_sidebar(vector_store)

    render_page_header()
    render_chat(rag_chain)


if __name__ == "__main__":
    main()