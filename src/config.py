import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_DIR   = os.path.join(BASE_DIR, "db")
    DATA_DIR = os.path.join(BASE_DIR, "data")

    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    LLM_TYPE       = "ollama"
    LLM_MODEL      = "llama3.2"
    OLLAMA_BASE_URL = "http://localhost:11434"

    # Збільшено: 250→500 щоб визначення не розривалось між чанками
    CHUNK_SIZE    = 500
    CHUNK_OVERLAP = 100   # більший overlap = менше втрат на межах

    # Збільшено: 6→8 щоб покрити більше контексту при пошуку
    K_RETRIEVAL = 8

for directory in [Config.DB_DIR, Config.DATA_DIR]:
    os.makedirs(directory, exist_ok=True)