"""
scan_all_topics.py — повний скан тем в Python Programming.pdf
щоб зрозуміти що реально є в книзі Halvorsen.

Запуск:
    python scan_all_topics.py
"""
from src.vector_store import VectorStore

vs = VectorStore()
all_docs = vs._db.get(where={"source": "Python Programming.pdf"})
texts = all_docs.get("documents", [])
metas = all_docs.get("metadatas", [])

print(f"Чанків з Python Programming.pdf: {len(texts)}\n")

TOPICS = {
    # Базові типи
    "lists":              ["mylist", "list()", "append(", "list = [", ".append", "lists in python"],
    "dictionary":         ["dictionary", "mydict", "dict()", ".keys()", ".values()", ".items()"],
    "strings":            ["string", "str()", ".upper()", ".lower()", ".split()", "string method"],
    "numbers/math":       ["integer", "float", "math.sqrt", "numpy", "arithmetic"],

    # Контроль потоку
    "if/else":            ["if ", "else:", "elif ", "conditional"],
    "for loop":           ["for i in", "for x in", "range(", "for loop"],
    "while loop":         ["while ", "while loop"],

    # Функції
    "functions":          ["def ", "def function", "return ", "parameter", "argument"],
    "recursion":          ["recursion", "recursive", "factorial"],

    # ООП
    "classes/OOP":        ["class ", "__init__", "self.", "object", "inheritance"],

    # Файли
    "file I/O":           ["open(", "read()", "write(", "file", ".txt", "readline"],

    # Числові бібліотеки
    "numpy":              ["numpy", "np.array", "np.zeros", "ndarray"],
    "matplotlib":         ["matplotlib", "plt.plot", "pyplot", "plt.show"],
    "scipy":              ["scipy", "from scipy"],

    # Модулі
    "modules/import":     ["import ", "from ", "module", "library"],

    # Виключення
    "exceptions":         ["try:", "except ", "raise ", "exception"],
}

print(f"{'ТЕМА':<25} {'ЧАНКІВ':>8}   ПРИКЛАД")
print("─" * 75)

for topic, keywords in TOPICS.items():
    found = []
    for text, meta in zip(texts, metas):
        t = text.lower()
        if any(kw.lower() in t for kw in keywords):
            found.append((meta.get("page", "?"), text))

    status = f"{len(found):>6}" if found else "     ❌"
    example = ""
    if found:
        # Беремо найрелевантніший чанк (де keyword найближче до початку)
        example = found[0][1][:80].replace("\n", " ")

    print(f"{topic:<25} {status}   {example}")