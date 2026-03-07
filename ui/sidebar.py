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

MAX_FILE_SIZE_MB = 50


def render_sidebar(vector_store) -> None:
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

        if process_btn:
            _handle_process(uploaded_files, vector_store)

        if clear_btn:
            _handle_clear(vector_store)

        # ── Статистика ────────────────────────────────────────
        section_label("Статистика")
        render_stats(
            doc_count=len(st.session_state.get("doc_names", [])),
            chunk_count=st.session_state.get("total_chunks", 0),
        )

        if st.session_state.get("doc_names"):
            section_label("Завантажено")
            render_doc_list(st.session_state["doc_names"])


# ── Приватні хелпери ──────────────────────────────────────────────────────────

def _handle_process(uploaded_files, vector_store) -> None:
    from src.document_loader import load_and_split_documents

    if not uploaded_files:
        st.warning("Спочатку оберіть файли.")
        return

    bar = st.progress(0, text="")
    processed_docs = st.session_state.get("doc_names", [])
    any_added = False

    for i, file in enumerate(uploaded_files):
        bar.progress((i + 0.5) / len(uploaded_files), text=f"⟳  {file.name}")

        # Перевірка: вже оброблено
        if file.name in processed_docs:
            continue

        # Перевірка розміру файлу
        if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"Файл «{file.name}» перевищує {MAX_FILE_SIZE_MB} MB — пропущено.")
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

            # Виправлення #7: додаємо до doc_names тільки після успішного
            # збереження у векторну базу — уникаємо неконсистентного стану.
            vector_store.add_documents(docs)
            st.session_state.setdefault("doc_names", []).append(file.name)
            any_added = True

        except Exception as e:
            st.error(f"Помилка з файлом {file.name}: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    bar.empty()

    if any_added:
        # Виправлення #2: беремо реальну кількість з бази, а не додаємо до
        # попереднього значення session_state — гарантує консистентність.
        st.session_state["total_chunks"] = vector_store.count()
        st.toast(f"✓ Документи додано. Фрагментів у базі: {st.session_state['total_chunks']}")
    else:
        st.toast("ℹ️ Усі вибрані файли вже були оброблені раніше.")


def _handle_clear(vector_store) -> None:
    vector_store.clear()
    st.session_state["messages"]     = []
    st.session_state["doc_names"]    = []
    st.session_state["total_chunks"] = 0
    # Скидаємо кеш після того як база вже фізично видалена з диска.
    # Порядок важливий: спочатку clear() (видаляє файли + перестворює _db),
    # потім cache_resource.clear() (змушує Streamlit перестворити VectorStore).
    st.cache_resource.clear()
    st.rerun()