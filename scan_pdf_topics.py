"""
scan_pdf_topics.py — сканує всі чанки з Python Programming.pdf
і шукає конкретні теми по ключових словах.

Запуск:
    python scan_pdf_topics.py
"""
from src.vector_store import VectorStore

vs = VectorStore()

# Отримуємо ВСІ документи з колекції
all_docs = vs._db.get(where={"source": "Python Programming.pdf"})

texts = all_docs.get("documents", [])
metas = all_docs.get("metadatas", [])

print(f"Всього чанків з Python Programming.pdf: {len(texts)}\n")

TOPICS = {
    "dictionary / dict": ["dict", "dictionary", "{}", "key", "value"],
    "tuple":             ["tuple", "(1, 2", "immutable"],
    "list comprehension":["comprehension", "[x for", "list comp"],
    "lambda":            ["lambda", "anonymous function"],
    "with statement":    ["with open", "with statement", "context manager", "__enter__"],
}

for topic, keywords in TOPICS.items():
    found = []
    for text, meta in zip(texts, metas):
        t = text.lower()
        if any(kw.lower() in t for kw in keywords):
            found.append((meta.get("page", "?"), text[:200]))

    print(f"{'─'*60}")
    print(f"ТЕМА: {topic} — знайдено {len(found)} чанків")
    for page, snippet in found[:3]:  # показуємо до 3
        print(f"  p.{page}: {snippet[:150]}")
    if not found:
        print("  ❌ ТЕМА ВІДСУТНЯ В ДОКУМЕНТІ")
    print()