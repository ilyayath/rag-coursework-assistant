"""
ui/sidebar.py — логіка бічної панелі:
завантаження файлів, обробка, очищення бази.
"""
import os
import tempfile
import streamlit as st

from ui.components import (
    render_sidebar_logo,
    section_label,
    render_stats,
    render_doc_list,
)


def render_sidebar(vector_store) -> None:
    """
    Рендерить бічну панель і виконує всю логіку
    завантаження / обробки / очищення документів.
    """
    with st.sidebar:
        render_sidebar_logo()

        # ── Завантаження файлів ───────────────────────────────
        section_label("Документи", margin_top=0)
        uploaded_files = st.file_uploader(
            "Перетягніть або оберіть файли",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        col1, col2 = st.columns(2)
        process_btn = col1.button("▶ Обробити", use_container_width=True)
        clear_btn   = col2.button("✕ Очистити", use_container_width=True)

        # ── Обробка ───────────────────────────────────────────
        if process_btn:
            _handle_process(uploaded_files, vector_store)

        # ── Очищення ──────────────────────────────────────────
        if clear_btn:
            _handle_clear(vector_store)

        # ── Статистика ────────────────────────────────────────
        section_label("Статистика")
        render_stats(
            doc_count=len(st.session_state.get("doc_names", [])),
            chunk_count=st.session_state.get("total_chunks", 0),
        )

        # ── Список файлів ─────────────────────────────────────
        if st.session_state.get("doc_names"):
            section_label("Завантажено")
            render_doc_list(st.session_state["doc_names"])


# ── Приватні хелпери ──────────────────────────────────────────────────────────

def _handle_process(uploaded_files, vector_store) -> None:
    from src.document_loader import load_and_split_documents
    import os
    import tempfile

    if not uploaded_files:
        st.warning("Спочатку оберіть файли.")
        return

    bar = st.progress(0, text="")
    new_chunks = 0

    # Отримуємо список вже оброблених файлів
    processed_docs = st.session_state.get("doc_names", [])

    for i, file in enumerate(uploaded_files):
        progress = (i + 0.5) / len(uploaded_files)
        bar.progress(progress, text=f"⟳  {file.name}")

        # ПЕРЕВІРКА: Якщо файл вже в базі — пропускаємо його
        if file.name in processed_docs:
            continue

        tmp_path = None
        try:
            suffix = "." + file.name.rsplit(".", 1)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name

            docs = load_and_split_documents(tmp_path)
            for doc in docs:
                doc.metadata["source"] = file.name

            vector_store.add_documents(docs)
            new_chunks += len(docs)

            # Додаємо файл до списку оброблених, щоб наступного разу його пропустити
            st.session_state.setdefault("doc_names", []).append(file.name)

        except Exception as e:
            st.error(f"Помилка з файлом {file.name}: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # Оновлюємо лічильник чанків тільки якщо додалися нові
    if new_chunks > 0:
        st.session_state["total_chunks"] = st.session_state.get("total_chunks", 0) + new_chunks
        st.toast(f"✓ Додано {new_chunks} нових фрагментів")
    else:
        st.toast("ℹ️ Усі вибрані файли вже були оброблені раніше.")

    bar.empty()


def _handle_clear(vector_store) -> None:
    vector_store.clear()
    st.session_state["messages"]     = []
    st.session_state["doc_names"]    = []
    st.session_state["total_chunks"] = 0
    st.rerun()