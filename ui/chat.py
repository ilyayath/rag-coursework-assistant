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

    if prompt := st.chat_input("Введіть запит до документів…"):
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

            render_sources(sources)

            messages.append({
                "role":    "assistant",
                "content": full_answer,
                "sources": sources,
            })

        except Exception as e:
            st.error(f"Помилка генерації: {e}")