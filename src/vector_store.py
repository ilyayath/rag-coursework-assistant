import os
import shutil
import gc
import time
from typing import List, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.config import Config
from src.logger import get_logger

logger = get_logger("VectorStore")


class VectorStore:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL_NAME
        )
        self.persist_directory = Config.DB_DIR

    def add_documents(self, documents: List[Document]):
        if not documents:
            logger.warning("Спроба додати порожній список документів.")
            return

        logger.info(f"Додаю {len(documents)} фрагментів у базу...")

        vector_db = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory
        )

        vector_db.add_documents(documents)

        # Явне видалення об'єкта для коректного звільнення ресурсів
        del vector_db
        gc.collect()

        logger.info("Документи успішно збережені.")

    def search_with_score(self, query: str, k: int = Config.K_RETRIEVAL) -> List[Tuple[Document, float]]:
        vector_db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )

        # ВИМИКАЄМО MMR, бо тепер у нас є справжній Reranker!
        # Використовуємо звичайний пошук, який знайде "Times New Roman"
        # і поверне реальні дистанції.
        results_with_scores = vector_db.similarity_search_with_score(query, k=k)

        del vector_db
        gc.collect()

        return results_with_scores

    def clear(self):
        gc.collect()
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                os.makedirs(self.persist_directory)
                logger.info("База даних очищена.")
            except PermissionError:
                logger.warning("Windows блокує файли. Повторна спроба через 1с...")
                time.sleep(1)
                try:
                    shutil.rmtree(self.persist_directory)
                    os.makedirs(self.persist_directory)
                    logger.info("База даних очищена (спроба 2).")
                except Exception as e:
                    logger.error(f"Критична помилка очищення: {e}")
        else:
            logger.info("База даних вже порожня.")