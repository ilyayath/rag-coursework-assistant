from src.vector_store import VectorStore

vs = VectorStore()

# Пробуємо різні варіанти запиту
queries = [
    "шрифт",
    "Times New Roman",
    "розмір шрифту",
    "оформлення тексту",
    "14 кегль"
]

for q in queries:
    print(f"\n=== Запит: '{q}' ===")
    results = vs.search_with_score(q, k=3)
    for doc, score in results:
        print(f"Score: {score:.4f} | {doc.page_content[:200]}")