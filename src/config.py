import os
from dotenv import load_dotenv

# Завантажуємо змінні оточення (якщо будуть)
load_dotenv()

class Config:
    # --- ШЛЯХИ ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_DIR = os.path.join(BASE_DIR, "db")
    DATA_DIR = os.path.join(BASE_DIR, "data")

    # --- МОДЕЛІ ---
    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    # Налаштування LLM (Ollama)
    LLM_TYPE = "ollama"
    LLM_MODEL = "llama3.2"
    OLLAMA_BASE_URL = "http://localhost:11434"

    # --- RAG ПАРАМЕТРИ ---
    CHUNK_SIZE = 250      # Синхронізовано з document_loader.py
    CHUNK_OVERLAP = 50
    # Зменшено з 15 до 5 — малі моделі краще працюють з меншим контекстом
    K_RETRIEVAL = 6

# Автоматичне створення папок, якщо їх немає
for directory in [Config.DB_DIR, Config.DATA_DIR]:
    os.makedirs(directory, exist_ok=True)