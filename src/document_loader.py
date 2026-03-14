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
    """
    text = text.replace('\xa0', ' ').replace('\t', ' ')
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ ]{2,}', ' ', text)
    return text.strip()


def load_and_split_documents(file_path: str) -> List[Document]:
    """
    Завантажує файл. Підтримувані формати: PDF, TXT, DOCX, MD.
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
            for doc in documents:
                doc.page_content = clean_text(doc.page_content)
                doc.metadata.setdefault("page", 1)

        elif file_extension == ".md":
            # Markdown — завантажуємо як текст, зберігаємо структуру заголовків
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
            for doc in documents:
                doc.page_content = clean_text(doc.page_content)
                doc.metadata.setdefault("page", 1)

        elif file_extension == ".docx":
            try:
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(file_path)
                documents = loader.load()
                for doc in documents:
                    doc.page_content = clean_text(doc.page_content)
                    doc.metadata.setdefault("page", 1)
            except ImportError:
                logger.error(
                    "Для завантаження .docx встановіть: pip install docx2txt"
                )
                return []

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

        # Відфільтровуємо надто короткі фрагменти — вони засмічують базу
        # і погіршують якість пошуку (порожні сторінки PDF, заголовки тощо).
        MIN_CHUNK_LEN = 30
        before = len(split_docs)
        split_docs = [d for d in split_docs if len(d.page_content.strip()) >= MIN_CHUNK_LEN]
        if len(split_docs) < before:
            logger.info(f"Відфільтровано {before - len(split_docs)} коротких фрагментів (<{MIN_CHUNK_LEN} символів).")

        for doc in split_docs:
            doc.metadata["source"] = os.path.basename(
                doc.metadata.get("source", "unknown")
            )

        logger.info(f"Завантажено {len(split_docs)} фрагментів.")
        return split_docs

    except Exception as e:
        logger.error(f"Помилка при обробці файлу {file_path}: {e}")
        return []