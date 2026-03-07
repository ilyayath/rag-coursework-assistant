import os
import re
from typing import List
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import Config
from src.logger import get_logger

logger = get_logger("DocumentLoader")


def clean_text(text: str) -> str:
    """
    Очищує текст після PDF, зберігаючи структуру списків і таблиць.

    Зміни відносно попередньої версії:
    - Одиночні \\n більше НЕ замінюються на пробіл —
      це ламало списки, нумеровані пункти і таблиці.
    - Замість цього стискаємо лише зайві пробіли в межах рядка.
    - Три і більше послідовних \\n зводимо до двох (порожній рядок).
    """
    # Нормалізуємо різні види пробільних символів
    text = text.replace('\xa0', ' ').replace('\t', ' ')

    # Стискаємо 3+ переходи рядка до двох (зберігаємо абзаци)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Стискаємо зайві пробіли ВСЕРЕДИНІ рядка (але не самі \n)
    text = re.sub(r'[ ]{2,}', ' ', text)

    return text.strip()


def load_and_split_documents(file_path: str) -> List[Document]:
    """
    Завантажує файл, використовуючи pdfplumber для кращої якості.
    """
    documents = []
    file_extension = os.path.splitext(file_path)[1].lower()

    try:
        logger.info(f"Початок обробки файлу: {file_path}")

        if file_extension == ".pdf":
            loader = PDFPlumberLoader(file_path)
            raw_docs = loader.load()

            for doc in raw_docs:
                doc.page_content = clean_text(doc.page_content)
                if "page" not in doc.metadata:
                    doc.metadata["page"] = 1

            documents = raw_docs

        elif file_extension == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
        else:
            logger.error(f"Формат {file_extension} не підтримується.")
            return []

        if not documents:
            logger.warning("Файл порожній або не вдалося зчитати текст.")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        split_docs = text_splitter.split_documents(documents)

        for doc in split_docs:
            doc.metadata["source"] = os.path.basename(
                doc.metadata.get("source", "unknown")
            )

        logger.info(f"Завантажено {len(split_docs)} фрагментів.")
        return split_docs

    except Exception as e:
        logger.error(f"Помилка при обробці файлу {file_path}: {e}")
        return []