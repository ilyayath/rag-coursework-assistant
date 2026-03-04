"""
diagnose_chunks.py — показує які чанки реально знаходить Chroma
для проблемних запитів.

Запуск:
    python diagnose_chunks.py
"""
from src.vector_store import VectorStore

vs = VectorStore()

FAILING_QUERIES = [
    "What is a dictionary in Python?",
    "What is list comprehension in Python?",
    "What is a tuple in Python?",
    "What is a lambda function in Python?",
    "How do you use the with statement in Python?",
]

for query in FAILING_QUERIES:
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print('='*70)
    results = vs.search_with_score(query, k=4)
    if not results:
        print("  ⚠️  Нічого не знайдено!")
        continue
    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get("source", "?")
        page   = doc.metadata.get("page", "?")
        print(f"\n  Rank {i+1} | score={score:.4f} | {source} p.{page}")
        print(f"  {doc.page_content[:300]}")