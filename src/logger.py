import logging
import sys


def get_logger(name: str):
    """
    Налаштовує та повертає логер із заданим іменем.
    Формат виводу: [Час] [Рівень] [Модуль]: Повідомлення
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # Вимикаємо передачу до кореневого логера, щоб уникнути дублювання
        logger.propagate = False

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger