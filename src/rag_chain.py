"""
src/rag_chain.py
"""
from typing import Dict, Any, List, Iterator

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import Config
from src.vector_store import VectorStore
from src.logger import get_logger

logger = get_logger("RAGChain")


class RAGChain:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

        if Config.LLM_TYPE == "ollama":
            self.llm = ChatOllama(
                model=Config.LLM_MODEL,
                base_url=Config.OLLAMA_BASE_URL,
                temperature=0.1,
            )
        else:
            self.llm = ChatOpenAI(
                model=Config.LLM_MODEL,
                temperature=0.1,
                api_key=Config.OPENAI_API_KEY,
            )

        template = """You are a document analysis assistant. You ONLY answer based on the CONTEXT below.

CONTEXT:
{context}

USER QUESTION:
{question}

RULES — follow them exactly, no exceptions:
1. Use ONLY the information in CONTEXT above.
2. Be direct. Do not start with "Based on the context..." or similar filler.
3. If the answer is not present in CONTEXT, you MUST output this exact sentence and nothing else:
   I cannot find the answer in the provided documents.
   Do NOT explain why. Do NOT suggest alternatives. Do NOT say "I cannot provide". Output ONLY that sentence.
4. Answer in the same language the USER QUESTION was asked in.

ANSWER:"""

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    # ── Спільна логіка retrieval ──────────────────────────────────────────────

    def _retrieve(self, query: str) -> tuple[List, List[Dict]]:
        """
        Виконує пошук, фільтрацію та будує context_parts і sources.
        Повертає (context_parts, sources) або (None, None) якщо нічого не знайдено.
        """
        results_with_scores = self.vector_store.search_with_score(
            query, k=Config.K_RETRIEVAL
        )

        if not results_with_scores:
            logger.warning("Не знайдено релевантних документів.")
            return None, None

        # Логуємо scores для діагностики — дивись консоль щоб підібрати
        # правильний SCORE_THRESHOLD для своєї моделі та даних.
        for i, (doc, score) in enumerate(results_with_scores):
            logger.info(
                f"Rank {i + 1} score={score:.4f} | "
                f"{doc.metadata.get('source','?')} p.{doc.metadata.get('page','?')} | "
                f"{doc.page_content[:80].replace(chr(10),' ')}..."
            )

        # Фільтрація за порогом — якщо всі чанки відфільтровані,
        # повертаємо топ-3 без фільтра щоб не залишати користувача без відповіді.
        filtered = [
            (doc, score)
            for doc, score in results_with_scores
            if score <= Config.SCORE_THRESHOLD
        ]

        if not filtered:
            logger.warning(
                f"Всі результати перевищують поріг {Config.SCORE_THRESHOLD}. "
                f"Використовую топ-3 без фільтра. "
                f"Розглянь збільшення SCORE_THRESHOLD у config.py."
            )
            filtered = results_with_scores[:3]

        logger.debug(
            f"Після фільтрації: {len(filtered)}/{len(results_with_scores)} чанків "
            f"(поріг {Config.SCORE_THRESHOLD})"
        )

        top_docs = [doc for doc, _ in filtered]

        for i, (doc, score) in enumerate(filtered):
            logger.debug(
                f"Rank {i + 1} (score={score:.4f}): {doc.page_content[:100]}..."
            )

        context_parts = []
        for doc in top_docs:
            source = doc.metadata.get("source", "unknown")
            page   = doc.metadata.get("page", "?")
            context_parts.append(
                f"--- Document: {source} (Page {page}) ---\n{doc.page_content}"
            )

        sources: List[Dict] = []
        seen: set = set()
        for doc in top_docs:
            sid = f"{doc.metadata.get('source')}_{doc.metadata.get('page')}"
            if sid not in seen:
                sources.append({
                    "source": doc.metadata.get("source", "unknown"),
                    "page":   doc.metadata.get("page", 1),
                })
                seen.add(sid)

        return context_parts, sources

    # ── Звичайний (не-streaming) режим ───────────────────────────────────────

    def ask(self, query: str) -> Dict[str, Any]:
        """Повертає повну відповідь одразу."""
        logger.debug(f"Query: {query}")

        context_parts, sources = self._retrieve(query)
        if context_parts is None:
            return {
                "answer": "I cannot find the answer in the provided documents.",
                "sources": [],
            }

        context_text = "\n\n".join(context_parts)
        logger.debug("Generating response...")
        response_text = self.chain.invoke({
            "context":  context_text,
            "question": query,
        })

        return {"answer": response_text, "sources": sources}

    # ── Streaming режим ───────────────────────────────────────────────────────

    def ask_stream(self, query: str) -> tuple[Iterator[str], List[Dict]]:
        """
        Повертає (stream, sources).
        stream — генератор рядків токенів для st.write_stream().
        sources — список джерел (готовий одразу, до початку стрімінгу).

        Використання в chat.py:
            stream, sources = rag_chain.ask_stream(query)
            st.write_stream(stream)
            render_sources(sources)
        """
        logger.debug(f"Stream query: {query}")

        context_parts, sources = self._retrieve(query)

        if context_parts is None:
            # Повертаємо генератор з єдиним повідомленням про відсутність відповіді
            def _not_found():
                yield "I cannot find the answer in the provided documents."
            return _not_found(), []

        context_text = "\n\n".join(context_parts)
        logger.debug("Streaming response...")

        # LangChain chain.stream() повертає ітератор рядків
        stream = self.chain.stream({
            "context":  context_text,
            "question": query,
        })

        return stream, sources