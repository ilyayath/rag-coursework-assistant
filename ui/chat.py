"""
ui/chat.py — логіка чат-панелі:
відображення історії та обробка нового запиту.
"""
import streamlit as st

from ui.components import render_empty_state, render_sources


def render_chat(rag_chain) -> None:
    """
    Рендерить всю чат-область:
    порожній стан → історія → input.
    """
    messages: list[dict] = st.session_state.get("messages", [])

    empty_placeholder = st.empty()

    if not messages:
        with empty_placeholder.container():
            render_empty_state()

    # Історія повідомлень
    for msg in messages:
        _render_message(msg)

    # Обробка нового запиту
    if prompt := st.chat_input("Введіть запит до документів…"):
        empty_placeholder.empty()
        _handle_user_input(prompt, rag_chain)


# ── Приватні хелпери ──────────────────────────────────────────────────────────

def _render_message(msg: dict) -> None:
    """Рендерить одне повідомлення з опціональним блоком джерел."""
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])


def _handle_user_input(prompt: str, rag_chain) -> None:
    """Додає повідомлення користувача, стрімить та відображає відповідь."""
    # Зберігаємо та показуємо запит
    st.session_state.setdefault("messages", []).append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Генеруємо відповідь зі стрімінгом
    with st.chat_message("assistant"):
        try:
            stream, sources = rag_chain.ask_stream(prompt)

            # st.write_stream() приймає будь-який ітератор рядків,
            # виводить токени поступово і повертає повний зібраний текст.
            full_answer = st.write_stream(stream)

            render_sources(sources)

            st.session_state["messages"].append({
                "role":    "assistant",
                "content": full_answer,
                "sources": sources,
            })

        except Exception as e:
            st.error(f"Помилка генерації: {e}")