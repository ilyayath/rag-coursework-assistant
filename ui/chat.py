"""
ui/chat.py — логіка чат-панелі з conversation history.
"""
import streamlit as st

from ui.components import render_empty_state, render_sources


def render_chat(rag_chain) -> None:
    messages: list[dict] = st.session_state.get("messages", [])

    empty_placeholder = st.empty()

    if not messages:
        with empty_placeholder.container():
            render_empty_state()

    for msg in messages:
        _render_message(msg)

    # Блокуємо введення якщо база порожня
    has_docs = st.session_state.get("total_chunks", 0) > 0
    if not has_docs:
        st.chat_input("Спочатку завантажте документи…", disabled=True)
    elif prompt := st.chat_input("Введіть запит до документів…"):
        empty_placeholder.empty()
        _handle_user_input(prompt, rag_chain)


def _render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])


def _handle_user_input(prompt: str, rag_chain) -> None:
    messages = st.session_state.setdefault("messages", [])

    # Зберігаємо та показуємо запит користувача
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Передаємо історію (без поточного запиту) для контексту
    history = messages[:-1]

    with st.chat_message("assistant"):
        try:
            stream, sources = rag_chain.ask_stream(prompt, history=history)
            full_answer = st.write_stream(stream)

            # Захист: write_stream може повернути порожній рядок при обриві стріму.
            # Зберігаємо повідомлення лише якщо відповідь не порожня —
            # порожнє повідомлення в history ламає наступний _format_history.
            if full_answer:
                render_sources(sources)
                messages.append({
                    "role":    "assistant",
                    "content": full_answer,
                    "sources": sources,
                })
            else:
                st.warning("Отримано порожню відповідь. Спробуйте ще раз.")

        except Exception as e:
            st.error(f"Помилка генерації: {e}")