"""
src/vector_store.py — обгортка над Chroma.
Екземпляр Chroma відкривається ОДИН РАЗ і живе разом з об'єктом VectorStore.
"""
import os
import shutil
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
        logger.info("Ініціалізація embedding-моделі...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL_NAME
        )
        self.persist_directory = Config.DB_DIR
        self._db = self._open_db()
        logger.info("VectorStore готовий.")

    def _open_db(self) -> Chroma:
        """Відкриває (або створює) Chroma-колекцію."""
        return Chroma(
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
        )

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            logger.warning("Спроба додати порожній список документів.")
            return
        logger.info(f"Додаю {len(documents)} фрагментів у базу...")
        self._db.add_documents(documents)
        logger.info("Документи успішно збережені.")

    def search_with_score(
        self, query: str, k: int = Config.K_RETRIEVAL
    ) -> List[Tuple[Document, float]]:
        return self._db.similarity_search_with_score(query, k=k)

    def count(self) -> int:
        """Повертає кількість фрагментів у базі."""
        try:
            return self._db._collection.count()
        except Exception:
            return 0

    def clear(self) -> None:
        """Видаляє всі дані з диска і перестворює чисту базу."""
        logger.info("Очищення бази даних...")

        # 1. Відв'язуємо поточний об'єкт — НЕ викликаємо delete_collection(),
        #    бо після цього UUID стає невалідним і наступний add_documents падає.
        self._db = None

        # 2. Видаляємо директорію з диска
        if os.path.exists(self.persist_directory):
            for attempt in range(2):
                try:
                    shutil.rmtree(self.persist_directory)
                    break
                except PermissionError:
                    if attempt == 0:
                        logger.warning("Файли заблоковані. Повторна спроба через 1с...")
                        time.sleep(1)
                    else:
                        logger.error("Не вдалося видалити директорію бази даних.")
                        return

        os.makedirs(self.persist_directory, exist_ok=True)

        # 3. Відкриваємо нову чисту колекцію — новий UUID, без артефактів
        self._db = self._open_db()
        logger.info("База даних очищена і перестворена.")