import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_DIR   = os.path.join(BASE_DIR, "db")
    DATA_DIR = os.path.join(BASE_DIR, "data")

    # Для англійських документів all-MiniLM-L6-v2 дає значно кращі scores
    # ніж paraphrase-multilingual (яка оптимізована для мультилінгвальних задач).
    # Типові L2 scores: 0.2–0.8 (дуже релевантно), 1.0–1.5 (релевантно), >2.0 (слабо).
    # Якщо потрібна підтримка української — поверніться до multilingual моделі.
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    LLM_TYPE        = os.getenv("LLM_TYPE", "ollama")
    LLM_MODEL       = os.getenv("LLM_MODEL", "llama3.2")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")

    CHUNK_SIZE    = 500
    CHUNK_OVERLAP = 100
    K_RETRIEVAL   = 8

    # Поріг відстані для фільтрації нерелевантних чанків.
    # all-MiniLM-L6-v2 + Chroma повертає L2 distance:
    # 0.0–0.8 = дуже релевантно, 0.8–1.5 = релевантно, >2.0 = нерелевантно.
    # Дивись логи (Rank N score=...) щоб підібрати під свої дані.
    SCORE_THRESHOLD = 5.0

    @classmethod
    def validate(cls) -> None:
        """Перевіряє обов'язкові змінні конфігурації при старті."""
        if cls.LLM_TYPE == "openai" and not cls.OPENAI_API_KEY:
            raise EnvironmentError(
                "LLM_TYPE=openai, але OPENAI_API_KEY не задано у .env"
            )


for directory in [Config.DB_DIR, Config.DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# Валідація при імпорті — падаємо одразу, а не під час першого запиту
Config.validate()