import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_DIR   = os.path.join(BASE_DIR, "db")
    DATA_DIR = os.path.join(BASE_DIR, "data")

    # Для англійських документів all-MiniLM-L6-v2 дає значно кращі scores
    # ніж paraphrase-multilingual (яка оптимізована для мультилінгвальних задач).
    # Якщо потрібна підтримка української — поверніться до multilingual моделі.
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    LLM_TYPE        = os.getenv("LLM_TYPE", "ollama")
    LLM_MODEL       = os.getenv("LLM_MODEL", "mistral")
    OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "120"))  # timeout для Ollama
    OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")

    # Retrieval: беремо більше кандидатів щоб reranker мав з чого вибирати
    CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    K_RETRIEVAL   = int(os.getenv("K_RETRIEVAL",   "12"))

    # Поріг відстані для фільтрації нерелевантних чанків.
    # all-MiniLM-L6-v2 + Chroma: 0.0–0.8 дуже релевантно, >2.0 нерелевантно.
    SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "2.0"))

    # Reranker (cross-encoder)
    RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
    RERANKER_MODEL   = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_TOP_N   = int(os.getenv("RERANKER_TOP_N", "5"))

    # Мінімальна довжина фрагмента після чанкінгу (символів)
    MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", "30"))

    # Максимальний розмір файлу для завантаження (МБ)
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

    # Conversation history — скільки останніх пар (user+assistant) включати в промпт
    CONVERSATION_HISTORY_TURNS = int(os.getenv("CONVERSATION_HISTORY_TURNS", "3"))

    @classmethod
    def validate(cls) -> None:
        """Перевіряє обов'язкові змінні конфігурації при старті."""
        if cls.LLM_TYPE == "openai" and not cls.OPENAI_API_KEY:
            raise EnvironmentError(
                "LLM_TYPE=openai, але OPENAI_API_KEY не задано у .env"
            )


for directory in [Config.DB_DIR, Config.DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

Config.validate()