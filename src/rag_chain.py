"""
src/rag_chain.py — RAG pipeline з reranker та conversation history.
"""
from typing import Dict, Any, List, Iterator

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import Config
from src.vector_store import VectorStore
from src.reranker import Reranker
from src.logger import get_logger

logger = get_logger("RAGChain")

_NOT_FOUND = "I cannot find the answer in the provided documents."


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

        self.reranker = Reranker(Config.RERANKER_MODEL) if Config.RERANKER_ENABLED else None

        template = """You are a strict document-only assistant. Your ONLY knowledge source is the CONTEXT below.

{history}CONTEXT:
{context}

USER QUESTION:
{question}

CRITICAL RULES:
1. Use ONLY facts explicitly present in CONTEXT. Your own knowledge does NOT exist.
2. Be direct. No filler phrases.
3. If the topic is not covered in CONTEXT — even if you know the answer from training —
   you MUST output this exact sentence and nothing else:
   I cannot find the answer in the provided documents.
   Do NOT add recommendations, explanations, or suggestions. ONLY that sentence.
   Do NOT start with "A decorator is not mentioned" or similar — output ONLY the exact sentence above.
4. Answer in the same language as the USER QUESTION.
5. Use CONVERSATION HISTORY only to resolve pronouns like "it", "they", "this".

EXAMPLES OF CORRECT BEHAVIOR:
Q: What is a decorator?  [decorators not in CONTEXT]
A: I cannot find the answer in the provided documents.

Q: What is a list?  [lists ARE in CONTEXT]
A: A list is a sequence of values...

ANSWER:"""

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["history", "context", "question"],
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

        # Query rewriting — переформульовує follow-up запити для кращого retrieval.
        # Використовує той самий LLM але окремий легкий промпт.
        rewrite_template = """Given the conversation history and a follow-up question, \
rewrite the follow-up question to be a standalone search query that includes all necessary context.
If the question is already standalone, return it unchanged.
Return ONLY the rewritten query, nothing else.

CONVERSATION HISTORY:
{history}

FOLLOW-UP QUESTION: {question}

STANDALONE QUERY:"""

        self.rewrite_prompt = PromptTemplate(
            template=rewrite_template,
            input_variables=["history", "question"],
        )
        self.rewrite_chain = self.rewrite_prompt | self.llm | StrOutputParser()

    # ── Conversation history ──────────────────────────────────────────────────

    @staticmethod
    def _format_history(messages: List[Dict]) -> str:
        if not messages:
            return ""

        turns = Config.CONVERSATION_HISTORY_TURNS
        history_msgs = [m for m in messages if m["role"] in ("user", "assistant")]
        recent = history_msgs[-(turns * 2):]

        if not recent:
            return ""

        lines = ["CONVERSATION HISTORY (use only to resolve references):"]
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            if len(content) > 300:
                content = content[:300] + "..."
            lines.append(f"{role}: {content}")
        lines.append("")

        return "\n".join(lines) + "\n"

    # ── Query rewriting ──────────────────────────────────────────────────────

    def _rewrite_query(self, query: str, history: List[Dict]) -> str:
        """
        Переформульовує follow-up запит у standalone query для кращого retrieval.
        Якщо history порожня — повертає оригінальний запит без змін.
        """
        if not history:
            return query

        history_text = self._format_history(history)
        if not history_text:
            return query

        try:
            rewritten = self.rewrite_chain.invoke({
                "history":  history_text,
                "question": query,
            }).strip()
            if rewritten and rewritten != query:
                logger.info(f"Query rewrite: '{query}' → '{rewritten}'")
            return rewritten or query
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e} — використовую оригінал")
            return query

    # ── Retrieval + rerank ────────────────────────────────────────────────────

    def _retrieve(self, query: str) -> tuple[List, List[Dict]] | tuple[None, None]:
        results_with_scores = self.vector_store.search_with_score(
            query, k=Config.K_RETRIEVAL
        )

        if not results_with_scores:
            logger.warning("Не знайдено релевантних документів.")
            return None, None

        for i, (doc, score) in enumerate(results_with_scores):
            logger.info(
                f"Rank {i + 1} score={score:.4f} | "
                f"{doc.metadata.get('source','?')} p.{doc.metadata.get('page','?')} | "
                f"{doc.page_content[:80].replace(chr(10),' ')}..."
            )

        filtered = [
            (doc, score)
            for doc, score in results_with_scores
            if score <= Config.SCORE_THRESHOLD
        ]

        if not filtered:
            logger.warning(
                f"Всі результати перевищують поріг {Config.SCORE_THRESHOLD}. "
                f"Використовую топ-3 без фільтра."
            )
            filtered = results_with_scores[:3]

        if self.reranker and self.reranker.enabled:
            filtered = self.reranker.rerank(query, filtered, top_n=Config.RERANKER_TOP_N)
            logger.info(f"Після rerank: {len(filtered)} чанків")

        top_docs = [doc for doc, _ in filtered]

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

    # ── Публічні методи ───────────────────────────────────────────────────────

    def ask(self, query: str, history: List[Dict] | None = None) -> Dict[str, Any]:
        search_query = self._rewrite_query(query, history or [])
        context_parts, sources = self._retrieve(search_query)
        if context_parts is None:
            return {"answer": _NOT_FOUND, "sources": []}

        response_text = self.chain.invoke({
            "history":  self._format_history(history or []),
            "context":  "\n\n".join(context_parts),
            "question": query,
        })

        return {"answer": response_text, "sources": sources}

    def ask_stream(
        self,
        query: str,
        history: List[Dict] | None = None,
    ) -> tuple[Iterator[str], List[Dict]]:
        search_query = self._rewrite_query(query, history or [])
        context_parts, sources = self._retrieve(search_query)

        if context_parts is None:
            def _not_found():
                yield _NOT_FOUND
            return _not_found(), []

        stream = self.chain.stream({
            "history":  self._format_history(history or []),
            "context":  "\n\n".join(context_parts),
            "question": query,
        })

        return stream, sources