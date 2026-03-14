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
        # Захист: якщо _db чомусь None — відновлюємо колекцію перед записом.
        if self._db is None:
            logger.warning("_db == None перед add_documents — відновлюємо колекцію.")
            self._db = self._open_db()

        # Дедуплікація: отримуємо вже існуючі source-імена з бази
        # щоб не додати той самий файл двічі при повторному завантаженні.
        try:
            # Використовуємо публічний API ChromaDB
            existing = self._db.get(include=["metadatas"])
            existing_sources = {
                m.get("source") for m in existing["metadatas"] if m.get("source")
            }
        except Exception:
            existing_sources = set()

        new_docs = [
            doc for doc in documents
            if doc.metadata.get("source") not in existing_sources
        ]

        if not new_docs:
            logger.warning("Всі документи вже присутні в базі — пропускаємо.")
            return

        if len(new_docs) < len(documents):
            logger.info(
                f"Дедуплікація: {len(documents) - len(new_docs)} фрагментів пропущено "
                f"(вже в базі), додаю {len(new_docs)}."
            )

        logger.info(f"Додаю {len(new_docs)} фрагментів у базу...")
        try:
            self._db.add_documents(new_docs)
            logger.info("Документи успішно збережені.")
        except Exception as e:
            logger.error(f"Помилка при збереженні документів: {e}")
            raise  # пробрасуємо вгору щоб sidebar показав повідомлення про помилку

    def search_with_score(
        self, query: str, k: int = Config.K_RETRIEVAL
    ) -> List[Tuple[Document, float]]:
        if self._db is None:
            logger.error("search_with_score викликано поки _db == None.")
            return []
        try:
            return self._db.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error(f"Помилка векторного пошуку: {e}")
            return []

    def count(self) -> int:
        """Повертає кількість фрагментів у базі."""
        if self._db is None:
            logger.warning("count() викликано поки _db == None.")
            return 0
        try:
            # Використовуємо публічний API замість приватного _collection
            return len(self._db.get()["ids"])
        except Exception as e:
            logger.error(f"Помилка при count(): {e}")
            return 0

    def clear(self) -> None:
        """Видаляє всі дані з диска і перестворює чисту базу."""
        logger.info("Очищення бази даних...")

        # 1. Відв'язуємо поточний об'єкт — НЕ викликаємо delete_collection(),
        #    бо після цього UUID стає невалідним і наступний add_documents падає.
        self._db = None

        # 2. Видаляємо директорію з диска
        deleted = False
        if os.path.exists(self.persist_directory):
            for attempt in range(2):
                try:
                    shutil.rmtree(self.persist_directory)
                    deleted = True
                    break
                except PermissionError:
                    if attempt == 0:
                        logger.warning("Файли заблоковані. Повторна спроба через 1с...")
                        time.sleep(1)
                    else:
                        # Виправлення: НЕ робимо return — _db має бути відновлена
                        # в будь-якому випадку, інакше наступний add_documents впаде.
                        logger.error("Не вдалося видалити директорію — база може містити старі дані.")
        else:
            deleted = True

        os.makedirs(self.persist_directory, exist_ok=True)

        # 3. Відкриваємо нову колекцію — завжди, навіть якщо rmtree не вдався.
        self._db = self._open_db()
        if deleted:
            logger.info("База даних очищена і перестворена.")
        else:
            logger.warning("База перестворена, але стара директорія могла не видалитись.")