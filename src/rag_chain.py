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
    def __init__(self):
        # Поріг відключено (ставимо високий), щоб модель бачила хоч щось
        self.score_threshold = 20.0

        if Config.LLM_TYPE == "ollama":
            self.llm = ChatOllama(
                model=Config.LLM_MODEL,
                base_url=Config.OLLAMA_BASE_URL,
                temperature=0.3  # Трохи підняли креативність, щоб вона не мовчала
            )
        else:
            self.llm = ChatOpenAI(
                model=Config.LLM_MODEL,
                temperature=0.3,
                api_key=Config.OPENAI_API_KEY
            )

        self.vector_store = VectorStore()

        # --- ГОЛОВНА ЗМІНА: Промпт англійською ---
        # Ми просимо модель думати англійською (вона так розумніша),
        # але відповідати українською.
        template = """
        You are a helpful assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 

        IMPORTANT: Answer in Ukrainian language only.

        Context:
        {context}

        Question: {question}

        Answer in Ukrainian:
        """

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        self.chain = (
                self.prompt
                | self.llm
                | StrOutputParser()
        )

    def ask(self, query: str) -> Dict[str, Any]:
        print(f"\n--- [DEBUG] ЗАПИТ: {query} ---")

        # 1. Пошук
        results_with_scores = self.vector_store.search_with_score(query, k=Config.K_RETRIEVAL)

        relevant_docs = []
        for doc, score in results_with_scores:
            # Виводимо score для контролю
            print(f"[DEBUG] Found Chunk (Score: {score:.4f})")

            if score < self.score_threshold:
                relevant_docs.append(doc)

        if not relevant_docs:
            return {
                "answer": "Не знайдено релевантних документів (спробуйте перефразувати).",
                "sources": []
            }

        # 2. Формування контексту
        context_text = "\n\n".join(doc.page_content for doc in relevant_docs)

        # --- ДІАГНОСТИКА: ЩО БАЧИТЬ МОДЕЛЬ? ---
        # Виводимо перші 500 символів тексту, який ми знайшли.
        # Якщо тут "сміття" або ієрогліфи - проблема в PDF, а не в моделі.
        print(f"\n[DEBUG] КОНТЕКСТ ДЛЯ МОДЕЛІ:\n{context_text[:500]}...\n")

        # 3. Генерація
        print("[DEBUG] Генерація відповіді...")
        response_text = self.chain.invoke({
            "context": context_text,
            "question": query
        })

        # 4. Джерела
        sources = []
        seen_sources = set()
        for doc in relevant_docs:
            source_id = f"{doc.metadata.get('source')}_{doc.metadata.get('page')}"
            if source_id not in seen_sources:
                sources.append({
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", 1)
                })
                seen_sources.add(source_id)

        return {
            "answer": response_text,
            "sources": sources
        }