"""
app.py — точка входу застосунку.
"""

# ── Page config — ОБОВ'ЯЗКОВО першим ─────────────────────────────────────────
import streamlit as st

st.set_page_config(
    page_title="DocMind · RAG Assistant",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Стилі ────────────────────────────────────────────────────────────────────
from ui.styles import inject_styles
inject_styles()

# ── Імпорт логіки ────────────────────────────────────────────────────────────
try:
    from src.vector_store import VectorStore
    from src.rag_chain    import RAGChain
    from src.logger       import get_logger
except ImportError as e:
    st.error(f"Помилка імпорту модулів src: {e}")
    st.stop()

# ── Імпорт UI-модулів ────────────────────────────────────────────────────────
from ui.components import render_page_header
from ui.sidebar    import render_sidebar
from ui.chat       import render_chat

logger = get_logger("App")


# ── Ініціалізація системних компонентів (кешується між ререндерами) ──────────
@st.cache_resource(show_spinner="Завантаження моделей…")
def _load_system():
    """
    Один VectorStore → передається в RAGChain.
    Embedding-модель завантажується рівно один раз.
    Spinner показується тільки під час реального завантаження.
    """
    vs = VectorStore()
    return vs, RAGChain(vector_store=vs)


# ── Ініціалізація стану сесії ────────────────────────────────────────────────
def _init_session(vector_store: VectorStore) -> None:
    defaults = {
        "messages":     [],
        "doc_names":    [],
        "total_chunks": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    # Синхронізація: якщо сторінку перезавантажили, але база непорожня —
    # відновлюємо лічильник чанків зі справжнього стану Chroma.
    # Але НЕ відновлюємо якщо базу щойно очистили (флаг just_cleared).
    if st.session_state.pop("just_cleared", False):
        # База щойно очищена — залишаємо total_chunks = 0, не відновлюємо.
        pass
    elif st.session_state["total_chunks"] == 0:
        real_count = vector_store.count()
        if real_count > 0:
            st.session_state["total_chunks"] = real_count
            logger.info(f"Відновлено {real_count} фрагментів зі збереженої бази.")


# ── Головна функція ──────────────────────────────────────────────────────────
def main() -> None:
    try:
        vector_store, rag_chain = _load_system()
    except Exception as e:
        st.error(f"Критична помилка запуску: {e}")
        st.info("Перевірте, чи запущено Ollama і встановлені всі залежності.")
        st.stop()

    _init_session(vector_store)

    render_sidebar(vector_store)
    render_page_header()
    render_chat(rag_chain)


if __name__ == "__main__":
    main()