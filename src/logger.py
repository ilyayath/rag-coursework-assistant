"""
src/logger.py — централізоване налаштування логування.
"""
import logging


def get_logger(name: str) -> logging.Logger:
    """
    Повертає logger з форматом: timestamp — рівень — модуль — повідомлення.
    propagate=False запобігає дублюванню записів у кореневому логері Streamlit.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s — %(levelname)-8s — %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Запобігаємо дублюванню: Streamlit має власний кореневий логер,
        # без цього кожен запис з'являвся б двічі.
        logger.propagate = False

    return logger