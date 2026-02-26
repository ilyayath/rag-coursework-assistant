from typing import Dict, Any
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
        # Поріг схожості (емпірично підібраний для all-MiniLM-L6-v2)
        # Чим менше значення, тим суворіший відбір. 1.3 - 1.5 - це "м'який" поріг.
        self.score_threshold = 1.4

        if Config.LLM_TYPE == "ollama":
            self.llm = ChatOllama(
                model=Config.LLM_MODEL,
                base_url=Config.OLLAMA_BASE_URL,
                temperature=0
            )
        else:
            self.llm = ChatOpenAI(
                model=Config.LLM_MODEL,
                temperature=0,
                api_key=Config.OPENAI_API_KEY
            )

        self.vector_store = VectorStore()

        # Покращений промпт з інструкцією "Chain of Thought" (думай крок за кроком)
        template = """
        Ти — аналітичний асистент. Твоє завдання — відповідати на запитання виключно на основі наданого контексту.

        Правила:
        1. Використовуй ТІЛЬКИ наданий контекст. Не додавай інформацію з власної пам'яті.
        2. Якщо в контексті немає відповіді, напиши: "Інформація відсутня в наданих документах".
        3. Відповідь має бути чіткою, структурованою та українською мовою.

        Контекст:
        {context}

        Запитання: {question}

        Відповідь:
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
        logger.info(f"Отримано запит: {query}")

        # 1. Пошук з оцінками
        results_with_scores = self.vector_store.search_with_score(query, k=Config.K_RETRIEVAL)

        # 2. Фільтрація за порогом (Thresholding)
        relevant_docs = []
        for doc, score in results_with_scores:
            logger.info(f"Знайдено фрагмент (Score: {score:.4f}): {doc.page_content[:50]}...")

            # У ChromaDB distance score: 0 = ідентично, > 1.5 = мало схоже
            if score < self.score_threshold:
                relevant_docs.append(doc)
            else:
                logger.warning(f"Фрагмент відкинуто через низьку релевантність (Score: {score:.4f})")

        # Якщо після фільтрації нічого не залишилось
        if not relevant_docs:
            logger.info("Не знайдено релевантних документів після фільтрації.")
            return {
                "answer": "У завантажених документах немає інформації, що відповідає вашому запиту (низька релевантність).",
                "sources": []
            }

        # 3. Формування контексту
        context_text = "\n\n".join(doc.page_content for doc in relevant_docs)

        # 4. Генерація
        logger.info("Генерація відповіді через LLM...")
        response_text = self.chain.invoke({
            "context": context_text,
            "question": query
        })

        # 5. Джерела
        sources = []
        seen_sources = set()
        for doc in relevant_docs:
            source_id = f"{doc.metadata.get('source')}_p{doc.metadata.get('page')}"
            if source_id not in seen_sources:
                sources.append({
                    "source": doc.metadata.get("source", "Невідомий файл"),
                    "page": doc.metadata.get("page", 1)
                })
                seen_sources.add(source_id)

        return {
            "answer": response_text,
            "sources": sources
        }