import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import Config


def load_and_split_documents(file_path: str) -> List[Document]:
    """
    Зчитує файл (PDF або TXT) та розбиває його на частини (chunks).
    Використовує параметри з Config.
    """
    documents = []
    file_extension = os.path.splitext(file_path)[1].lower()

    try:
        # 1. Завантаження залежно від типу файлу
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif file_extension == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
        else:
            print(f"[ERROR] Формат {file_extension} не підтримується.")
            return []

        # 2. Налаштування спліттера (беремо параметри з Config)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )

        # 3. Розбиття на чанки
        split_docs = text_splitter.split_documents(documents)

        # 4. Очищення метаданих
        for doc in split_docs:
            # Якщо це TXT, додаємо сторінку 1
            if "page" not in doc.metadata:
                doc.metadata["page"] = 1

            # Залишаємо тільки назву файлу, а не повний шлях
            doc.metadata["source"] = os.path.basename(doc.metadata.get("source", "unknown"))

        print(f"[INFO] Завантажено {len(split_docs)} фрагментів з {file_path}")
        return split_docs

    except Exception as e:
        print(f"[ERROR] Помилка при обробці файлу {file_path}: {e}")
        return []