import os
from src.document_loader import load_and_split_documents
from src.vector_store import VectorStore
from src.rag_chain import RAGChain


def main():
    # Створюємо тестовий файл
    test_file = "sample_data.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Python — це високорівнева мова програмування загального призначення.\n")
        f.write("Створена Гвідо ван Россумом і вперше випущена у 1991 році.\n")
        f.write("Python підтримує структурне, об'єктно-орієнтоване та функціональне програмування.\n")
        f.write(
            "RAG (Retrieval-Augmented Generation) — це техніка, що покращує точність моделей LLM за допомогою зовнішніх даних.")

    print("[INFO] Початок інтеграційного тесту...")

    # 1. Завантаження
    print(f"[INFO] Завантаження файлу {test_file}...")
    docs = load_and_split_documents(test_file)

    # 2. Векторизація
    print("[INFO] Оновлення векторної бази...")
    vector_store = VectorStore()
    vector_store.clear()  # Починаємо з чистого листа
    vector_store.add_documents(docs)

    # 3. Ініціалізація RAG
    print("[INFO] Ініціалізація RAG ланцюжка...")
    rag = RAGChain()

    # 4. Тестовий запит
    question = "Яка погода в Лондоні?"
    print(f"[INFO] Запитання: {question}")

    result = rag.ask(question)

    print("-" * 50)
    print(f"[RESULT] Відповідь: {result['answer']}")
    print("-" * 50)
    print("[RESULT] Джерела:")
    for source in result['sources']:
        print(f" - Файл: {source['source']}, Сторінка: {source['page']}")
    print("-" * 50)

    # Прибирання
    if os.path.exists(test_file):
        os.remove(test_file)


if __name__ == "__main__":
    main()