"""
src/rag_chain.py
"""
from typing import Dict, Any, List

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

    def ask(self, query: str) -> Dict[str, Any]:
        logger.debug(f"Query: {query}")

        results_with_scores = self.vector_store.search_with_score(
            query, k=Config.K_RETRIEVAL
        )

        if not results_with_scores:
            logger.warning("Не знайдено релевантних документів.")
            return {
                "answer": "I cannot find the answer in the provided documents.",
                "sources": [],
            }

        top_docs = [doc for doc, _ in results_with_scores]

        for i, doc in enumerate(top_docs):
            logger.debug(f"Rank {i + 1}: {doc.page_content[:100]}...")

        context_parts = []
        for doc in top_docs:
            source = doc.metadata.get("source", "unknown")
            page   = doc.metadata.get("page", "?")
            context_parts.append(
                f"--- Document: {source} (Page {page}) ---\n{doc.page_content}"
            )
        context_text = "\n\n".join(context_parts)

        logger.debug("Generating response...")
        response_text = self.chain.invoke({
            "context":  context_text,
            "question": query,
        })

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

        return {"answer": response_text, "sources": sources}