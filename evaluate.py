"""
evaluate.py — автоматичне тестування RAG-системи.

Що вимірює:
  - L2_no_rerank  : евклідова відстань топ-1 фрагмента БЕЗ реранкінгу
  - CE_rerank     : оцінка крос-енкодера топ-1 фрагмента ПІСЛЯ реранкінгу
  - Faithfulness  : частка речень відповіді підкріплена контекстом (0.0–1.0)
  - Has_answer    : чи знайдена відповідь (1) чи «I cannot find...» (0)

Запуск:
    python evaluate.py

Результат зберігається у evaluation_results.csv та друкується у термінал.
"""

import csv
import sys
import time
from pathlib import Path

# ── Перевірка залежностей ──────────────────────────────────────────────────────
try:
    from sentence_transformers import CrossEncoder
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
except ImportError as e:
    print(f"[ERROR] Відсутня залежність: {e}")
    print("Встановіть: pip install -r requirements.txt")
    sys.exit(1)

# ── Додаємо кореневу директорію проєкту до шляху ──────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.vector_store import VectorStore
from src.reranker import Reranker
from src.rag_chain import RAGChain

# ══════════════════════════════════════════════════════════════════════════════
# НАБІР ТЕСТОВИХ ЗАПИТАНЬ
# Формат: (запитання, очікується_відповідь: True/False)
# True  = відповідь має бути знайдена в документах
# False = система має відмовити («I cannot find...»)
# ══════════════════════════════════════════════════════════════════════════════
TEST_QUESTIONS = [
    # ── Think Python ──────────────────────────────────────────────────────────
    ("What is a list in Python?",                   True),
    ("What methods does a Python list have?",        True),
    ("How does a for loop work in Python?",          True),
    ("What is a dictionary in Python?",              True),
    ("What is the difference between a tuple and a list?", True),
    # ── Pro Git ───────────────────────────────────────────────────────────────
    ("What is the difference between git merge and git rebase?", True),
    ("How do you create a new branch in Git?",       True),
    ("What does git stash do?",                      True),
    ("How do you undo the last commit in Git?",      True),
    ("What is a pull request?",                      True),
    # ── Поза межами документів ────────────────────────────────────────────────
    ("What is the capital of France?",               False),
    ("Who invented the telephone?",                  False),
    ("What is the speed of light?",                  False),
    ("Who wrote Hamlet?",                            False),
    ("What is the boiling point of water?",          False),
]

NOT_FOUND_PHRASE = "I cannot find the answer in the provided documents"


# ══════════════════════════════════════════════════════════════════════════════
# FAITHFULNESS
# Проста реалізація без зовнішнього LLM-суддівства:
# ділимо відповідь на речення і перевіряємо чи кожне речення
# семантично схоже на хоча б один фрагмент контексту.
# Поріг схожості: cosine similarity > 0.55
# ══════════════════════════════════════════════════════════════════════════════

def compute_faithfulness(answer: str, context_docs: list, embedder) -> float:
    """
    Повертає частку речень відповіді що підкріплені контекстом (0.0–1.0).
    Якщо відповідь порожня або «не знайдено» — повертає None.
    """
    if not answer or NOT_FOUND_PHRASE.lower() in answer.lower():
        return None

    # Ділимо на речення
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if len(s.strip()) > 10]
    if not sentences:
        return None

    context_texts = [doc.page_content for doc in context_docs]
    if not context_texts:
        return None

    # Векторизуємо речення та контекст
    sent_vecs = embedder.embed_documents(sentences)
    ctx_vecs  = embedder.embed_documents(context_texts)

    import numpy as np
    sent_vecs = np.array(sent_vecs)
    ctx_vecs  = np.array(ctx_vecs)

    # Нормалізуємо для косинусної схожості
    sent_norms = np.linalg.norm(sent_vecs, axis=1, keepdims=True)
    ctx_norms  = np.linalg.norm(ctx_vecs,  axis=1, keepdims=True)
    sent_vecs  = sent_vecs / (sent_norms + 1e-9)
    ctx_vecs   = ctx_vecs  / (ctx_norms  + 1e-9)

    # Для кожного речення — максимальна схожість з будь-яким фрагментом
    similarity_matrix = sent_vecs @ ctx_vecs.T  # (n_sent, n_ctx)
    max_similarities  = similarity_matrix.max(axis=1)

    THRESHOLD = 0.55
    supported = (max_similarities >= THRESHOLD).sum()
    return round(float(supported) / len(sentences), 3)


# ══════════════════════════════════════════════════════════════════════════════
# ГОЛОВНА ФУНКЦІЯ
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("RAG EVALUATION — автоматичне тестування системи")
    print("=" * 70)

    # ── Ініціалізація компонентів ──────────────────────────────────────────────
    print("\n[1/4] Завантаження embedding-моделі...")
    vector_store = VectorStore()

    # Перевіряємо що база не порожня
    count = vector_store.count()
    if count == 0:
        print("[ERROR] База даних порожня. Спочатку завантажте документи через UI.")
        sys.exit(1)
    print(f"       База: {count} фрагментів")

    print("[2/4] Завантаження крос-енкодера...")
    reranker = Reranker(Config.RERANKER_MODEL)
    if not reranker.enabled:
        print("[WARN]  Реранкер недоступний — CE score буде N/A")

    print("[3/4] Ініціалізація RAG-ланцюга...")
    rag_chain = RAGChain(vector_store=vector_store)

    # Пряме посилання на embedding-модель для Faithfulness
    embedder = vector_store.embedding_model

    print("[4/4] Починаємо тестування...\n")
    print("-" * 70)

    results = []

    for idx, (question, expects_answer) in enumerate(TEST_QUESTIONS, 1):
        print(f"[{idx:2d}/{len(TEST_QUESTIONS)}] {question[:60]}...")

        # ── Пошук БЕЗ реранкінгу ──────────────────────────────────────────────
        raw_results = vector_store.search_with_score(question, k=Config.K_RETRIEVAL)

        # Фільтрація за порогом
        filtered = [(doc, score) for doc, score in raw_results
                    if score <= Config.SCORE_THRESHOLD]
        if not filtered:
            filtered = raw_results[:3]

        l2_no_rerank = round(filtered[0][1], 4) if filtered else None

        # ── Реранкінг ─────────────────────────────────────────────────────────
        ce_score = None
        if reranker.enabled and filtered:
            reranked = reranker.rerank(question, filtered, top_n=Config.RERANKER_TOP_N)
            if reranked:
                ce_score = round(float(reranked[0][1]), 4)
            top_docs = [doc for doc, _ in reranked[:Config.RERANKER_TOP_N]]
        else:
            top_docs = [doc for doc, _ in filtered[:Config.RERANKER_TOP_N]]

        # ── Генерація відповіді ───────────────────────────────────────────────
        result = rag_chain.ask(question)
        answer = result.get("answer", "")

        # ── Has answer ────────────────────────────────────────────────────────
        has_answer = 0 if NOT_FOUND_PHRASE.lower() in answer.lower() else 1

        # Перевіряємо очікування
        correct = (has_answer == 1) == expects_answer
        status = "✅" if correct else "❌"

        # ── Faithfulness ──────────────────────────────────────────────────────
        faithfulness = compute_faithfulness(answer, top_docs, embedder) if has_answer else None

        results.append({
            "Запитання":       question,
            "Очікує відповідь": "Так" if expects_answer else "Ні",
            "Знайдено":        "Так" if has_answer else "Ні",
            "Коректно":        "Так" if correct else "Ні",
            "L2 (без реранк.)": l2_no_rerank,
            "CE score":        ce_score,
            "Faithfulness":    faithfulness,
        })

        faith_str = f"{faithfulness:.3f}" if faithfulness is not None else "—"
        ce_str    = f"{ce_score:.3f}"     if ce_score    is not None else "N/A"
        l2_str    = f"{l2_no_rerank:.4f}" if l2_no_rerank is not None else "N/A"

        print(f"       {status} L2={l2_str}  CE={ce_str}  Faith={faith_str}  "
              f"Відповідь: {'знайдена' if has_answer else 'не знайдена'}")

        # Невелика пауза щоб не перевантажити Ollama
        time.sleep(0.5)

    # ── Зберігаємо CSV ────────────────────────────────────────────────────────
    csv_path = ROOT / "evaluation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # ── Підсумкова статистика ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ПІДСУМОК")
    print("=" * 70)

    total    = len(results)
    correct  = sum(1 for r in results if r["Коректно"] == "Так")
    accuracy = correct / total * 100

    # Середні метрики тільки для запитань де відповідь знайдена
    with_answer = [r for r in results if r["Знайдено"] == "Так"]

    l2_vals    = [r["L2 (без реранк.)"] for r in with_answer if r["L2 (без реранк.)"] is not None]
    ce_vals    = [r["CE score"]         for r in with_answer if r["CE score"]         is not None]
    faith_vals = [r["Faithfulness"]     for r in with_answer if r["Faithfulness"]     is not None]

    avg_l2    = sum(l2_vals)    / len(l2_vals)    if l2_vals    else None
    avg_ce    = sum(ce_vals)    / len(ce_vals)    if ce_vals    else None
    avg_faith = sum(faith_vals) / len(faith_vals) if faith_vals else None

    print(f"\nТочність (коректних відповідей): {correct}/{total} = {accuracy:.1f}%")
    if avg_l2    is not None: print(f"Середня L2 відстань (без реранкінгу): {avg_l2:.4f}")
    if avg_ce    is not None: print(f"Середній CE score (з реранкінгом):     {avg_ce:.4f}")
    if avg_faith is not None: print(f"Середня Faithfulness:                  {avg_faith:.3f}")

    print(f"\nРезультати збережено: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()