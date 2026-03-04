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
        # Температура 0.1 для максимальної точності
        if Config.LLM_TYPE == "ollama":
            self.llm = ChatOllama(
                model=Config.LLM_MODEL,
                base_url=Config.OLLAMA_BASE_URL,
                temperature=0.1
            )
        else:
            self.llm = ChatOpenAI(
                model=Config.LLM_MODEL,
                temperature=0.1,
                api_key=Config.OPENAI_API_KEY
            )

        self.vector_store = VectorStore()

        # Повністю англомовний системний промпт.
        # Модель тепер "думає" у своєму природному середовищі.
        template = """
                You are a highly precise document analysis assistant. 

                CONTEXT:
                {context}

                USER QUESTION: 
                {question}

                STRICT INSTRUCTIONS:
                1. Answer the USER QUESTION using ONLY the provided CONTEXT.
                2. Be direct and concise. DO NOT start your response with filler phrases like "Based on the provided CONTEXT...". Just give the direct answer.
                3. Do not explain your reasoning. Output ONLY the final answer.
                4. If the exact answer is not available in the CONTEXT, strictly output this exact phrase and nothing else: "I cannot find the answer in the provided documents."
                5. Answer in English.

                ANSWER:
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
        print(f"\n--- [DEBUG] QUERY: {query} ---")

        # 1. Пошук у мультимовній базі
        results_with_scores = self.vector_store.search_with_score(query, k=5)

        if not results_with_scores:
            return {
                "answer": "No relevant documents found.",
                "sources": []
            }

        top_docs = [doc for doc, score in results_with_scores]

        print(f"\n[DEBUG] TOP CHUNKS FROM DB:")
        for i, doc in enumerate(top_docs):
            print(f"Rank {i + 1}: {doc.page_content[:100]}...")

        # 2. Формування контексту
        context_parts = []
        for doc in top_docs:
            source_name = doc.metadata.get('source', 'unknown')
            page_num = doc.metadata.get('page', '?')
            context_parts.append(f"--- Document: {source_name} (Page {page_num}) ---\n{doc.page_content}")

        context_text = "\n\n".join(context_parts)

        # 3. Генерація
        print("[DEBUG] Generating response...")
        response_text = self.chain.invoke({
            "context": context_text,
            "question": query
        })

        # 4. Збір джерел
        sources = []
        seen_sources = set()
        for doc in top_docs:
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