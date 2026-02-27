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
    Функція для очищення тексту від сміття після PDF.
    """
    # Заміна нерозривних пробілів та табуляцій
    text = text.replace('\xa0', ' ').replace('\t', ' ')
    # Видалення зайвих переносів рядків (коли речення розірване посередині)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Видалення множинних пробілів
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_and_split_documents(file_path: str) -> List[Document]:
    """
    Завантажує файл використовуючи pdfplumber для кращої якості.
    """
    documents = []
    file_extension = os.path.splitext(file_path)[1].lower()

    try:
        logger.info(f"Початок обробки файлу: {file_path}")

        if file_extension == ".pdf":
            # Використовуємо PDFPlumberLoader (він краще бачить таблиці і колонки)
            loader = PDFPlumberLoader(file_path)
            raw_docs = loader.load()

            # Очищаємо текст кожного фрагмента
            for doc in raw_docs:
                doc.page_content = clean_text(doc.page_content)
                # Додаємо метадані, якщо їх немає
                if "page" not in doc.metadata:
                    # pdfplumber зазвичай додає 'page', але про всяк випадок
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

        # Налаштування спліттера
        # Трохи збільшимо chunk_size, щоб захопити більше контексту
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Було 500, ставимо 800 для цілісності думок
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],  # Пріоритет розділювачів
            length_function=len
        )

        split_docs = text_splitter.split_documents(documents)

        # Фінальна підчистка метаданих
        for doc in split_docs:
            doc.metadata["source"] = os.path.basename(doc.metadata.get("source", "unknown"))

        logger.info(f"Завантажено {len(split_docs)} фрагментів.")
        return split_docs

    except Exception as e:
        logger.error(f"Помилка при обробці файлу {file_path}: {e}")
        return []