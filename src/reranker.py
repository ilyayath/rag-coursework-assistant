"""
src/reranker.py — cross-encoder reranker для покращення якості retrieval.

Після того як vector store повертає топ-K чанків за cosine/L2 distance,
reranker перераховує relevance score для кожної пари (query, chunk)
значно точніше — бо cross-encoder бачить обидва тексти одночасно.

Модель: cross-encoder/ms-marco-MiniLM-L-6-v2
- Розмір: ~90MB
- Швидкість: ~50ms на чанк на CPU, ~10ms на GPU
- Якість: значно краща за bi-encoder для re-ranking
"""
from typing import List, Tuple

from langchain_core.documents import Document
from src.logger import get_logger

logger = get_logger("Reranker")


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Завантаження reranker моделі: {model_name}...")
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name)
            self._enabled = True
            logger.info("Reranker готовий.")
        except ImportError:
            logger.warning(
                "sentence-transformers не встановлено — reranker вимкнено. "
                "Встановіть: pip install sentence-transformers"
            )
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def rerank(
        self,
        query: str,
        docs_with_scores: List[Tuple[Document, float]],
        top_n: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Перераховує scores для списку (Document, score) за допомогою cross-encoder.

        Повертає топ-N документів відсортованих за новим score (спадання).
        Якщо reranker вимкнено — повертає оригінальний список без змін.
        """
        if not self._enabled or not docs_with_scores:
            return docs_with_scores[:top_n]

        docs = [doc for doc, _ in docs_with_scores]

        # Cross-encoder оцінює кожну пару (query, chunk) — score вищий = релевантніший
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._model.predict(pairs)

        reranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        logger.debug(
            f"Reranker: {len(docs)} → топ-{top_n} | "
            f"кращий score: {reranked[0][1]:.4f} | "
            f"гірший з топу: {reranked[min(top_n-1, len(reranked)-1)][1]:.4f}"
        )

        # Повертаємо у форматі List[Tuple[Document, float]] (сумісно з vector_store)
        return [(doc, float(score)) for doc, score in reranked[:top_n]]