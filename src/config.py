import os
from dotenv import load_dotenv

# Завантажуємо змінні оточення (якщо будуть)
load_dotenv()

class Config:
    # --- ШЛЯХИ ---
    # Отримуємо абсолютний шлях до кореня проєкту
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Папка для бази даних (векторів)
    DB_DIR = os.path.join(BASE_DIR, "db")
    # Папка для завантажених файлів
    DATA_DIR = os.path.join(BASE_DIR, "data")

    # --- МОДЕЛІ ---
    # Модель для ембедингів (перетворення тексту в цифри)
    # Вона легка і працює навіть на CPU
    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Налаштування LLM (Ollama)
    LLM_TYPE = "ollama"
    LLM_MODEL = "llama3.2"  # Llama 3.2 (3B) за замовчуванням
    # Адреса локального сервера Ollama
    OLLAMA_BASE_URL = "http://localhost:11434"

    # --- RAG ПАРАМЕТРИ ---
    # Розмір шматка тексту (в символах)
    CHUNK_SIZE = 1000
    # Перекриття шматків для збереження контексту
    CHUNK_OVERLAP = 200
    # Кількість шматків, які ми шукаємо в базі (Context Window)
    # Беремо 4, щоб не перевантажувати маленьку модель зайвим текстом
    K_RETRIEVAL = 5

# Автоматичне створення папок, якщо їх немає
for directory in [Config.DB_DIR, Config.DATA_DIR]:
    os.makedirs(directory, exist_ok=True)