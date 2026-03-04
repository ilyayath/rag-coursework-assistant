"""
tests/test_rag_eval.py

Запуск:
    pytest tests/test_rag_eval.py -v --tb=short -s
"""

import time
import pytest
from src.vector_store import VectorStore
from src.rag_chain import RAGChain

MAX_LATENCY_SEC = 30

REFUSAL_PHRASES = [
    "cannot find the answer",
    "cannot provide",
    "i cannot",
    "not in the provided",
    "no information",
    "not found",
]


@pytest.fixture(scope="session")
def rag():
    vs = VectorStore()
    return RAGChain(vector_store=vs)


def ask(rag, question: str) -> tuple[str, float]:
    start = time.perf_counter()
    result = rag.ask(question)
    elapsed = time.perf_counter() - start
    return result["answer"].lower(), elapsed


def assert_contains(answer: str, keywords: list[str]) -> None:
    found = [kw for kw in keywords if kw.lower() in answer]
    assert found, (
        f"Відповідь не містить жодного з {keywords}.\nОтримано: {answer[:400]}"
    )


def assert_not_off_topic(answer: str) -> None:
    is_refusal = any(p in answer for p in REFUSAL_PHRASES)
    assert not is_refusal, (
        f"RAG відмовив на релевантне питання.\nВідповідь: {answer[:400]}"
    )


def assert_off_topic(answer: str) -> None:
    is_refusal = any(p in answer for p in REFUSAL_PHRASES)
    assert is_refusal, (
        f"RAG НЕ відмовив на off-topic питання.\nВідповідь: {answer[:400]}"
    )


# ══════════════════════════════════════════════════════════════
# БЛОК 1 — Python Basics (Halvorsen: практичний інженерний стиль)
# Теми підтверджені scan_all_topics.py
# ══════════════════════════════════════════════════════════════

class TestPythonBasics:

    def test_variables(self, rag):
        """50+ чанків про змінні та присвоєння."""
        answer, _ = ask(rag, "How do you create variables in Python?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["variable", "=", "assign", "value", "type"])

    def test_if_else(self, rag):
        """50 чанків про умовні оператори."""
        answer, _ = ask(rag, "How does an if-else statement work in Python?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["if", "else", "elif", "condition", "true", "false"])

    def test_for_loop(self, rag):
        """33 чанки про for-цикли."""
        answer, _ = ask(rag, "How do you use a for loop in Python?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["for", "range", "loop", "iterate", "in"])

    def test_while_loop(self, rag):
        """14 чанків про while-цикли."""
        answer, _ = ask(rag, "How does a while loop work in Python?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["while", "loop", "condition", "true"])

    def test_functions(self, rag):
        """34 чанки про функції."""
        answer, _ = ask(rag, "How do you define a function in Python?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["def", "function", "return", "parameter"])

    def test_function_multiple_return(self, rag):
        """Halvorsen має окремий розділ '6.2 Functions with multiple return values'."""
        answer, _ = ask(rag, "Can a Python function return multiple values?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["return", "multiple", "values", "function", "tuple"])

    def test_lists(self, rag):
        """7 чанків — append, len, for x in list."""
        answer, _ = ask(rag, "How do you work with lists in Python?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["list", "append", "len", "element", "index", "[]"])

    def test_string_basics(self, rag):
        """9 чанків про рядки."""
        answer, _ = ask(rag, "How do you work with strings in Python?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["string", "str", "print", "input", "text", "+"])

    def test_exception_handling(self, rag):
        """16 чанків — окремий розділ '10 Error Handling in Python'."""
        answer, _ = ask(rag, "How does error handling work in Python?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["try", "except", "error", "exception", "raise"])

    def test_oop_class(self, rag):
        """21 чанк про ООП і класи."""
        answer, _ = ask(rag, "How do you create a class in Python?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["class", "__init__", "self", "object", "instance"])


# ══════════════════════════════════════════════════════════════
# БЛОК 2 — Python: File I/O та Modules (сильна сторона Halvorsen)
# ══════════════════════════════════════════════════════════════

class TestPythonFileAndModules:

    def test_file_read(self, rag):
        """66 чанків про файловий I/O."""
        answer, _ = ask(rag, "How do you read a file in Python?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["open", "read", "file", "close", "readline", "with"])

    def test_file_write(self, rag):
        """Запис у файл."""
        answer, _ = ask(rag, "How do you write to a file in Python?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["open", "write", "file", "w", "close"])

    def test_import_module(self, rag):
        """159 чанків про import — найбільша тема в книзі."""
        answer, _ = ask(rag, "How do you import a module in Python?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["import", "module", "from", "library", "package"])


# ══════════════════════════════════════════════════════════════
# БЛОК 3 — Python: NumPy / Matplotlib / SciPy
# Це основна інженерна частина книги Halvorsen
# ══════════════════════════════════════════════════════════════

class TestPythonScientific:

    def test_numpy_array(self, rag):
        """29 чанків про NumPy."""
        answer, _ = ask(rag, "What is a NumPy array and how do you create one?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["numpy", "array", "np", "zeros", "ones", "ndarray"])

    def test_numpy_operations(self, rag):
        """Математичні операції з NumPy."""
        answer, _ = ask(rag, "How do you perform mathematical operations with NumPy?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["numpy", "np", "array", "math", "operation", "sqrt", "sum"])

    def test_matplotlib_plot(self, rag):
        """35 чанків про Matplotlib."""
        answer, _ = ask(rag, "How do you create a plot with Matplotlib?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["matplotlib", "plt", "plot", "show", "pyplot", "figure"])

    def test_scipy_usage(self, rag):
        """13 чанків про SciPy."""
        answer, _ = ask(rag, "What is SciPy used for in Python?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["scipy", "optimization", "linear", "integration", "scientific"])


# ══════════════════════════════════════════════════════════════
# БЛОК 4 — JavaScript Basics (Copes — повний довідник)
# ══════════════════════════════════════════════════════════════

class TestJavaScriptBasics:

    def test_var_let_const(self, rag):
        answer, _ = ask(rag, "What is the difference between var, let, and const in JavaScript?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["var", "let", "const", "scope", "block", "hoist"])

    def test_data_types(self, rag):
        answer, _ = ask(rag, "What are the primitive data types in JavaScript?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["string", "number", "boolean", "null", "undefined"])

    def test_array_methods(self, rag):
        answer, _ = ask(rag, "What methods can you use on arrays in JavaScript?")
        assert_not_off_topic(answer)
        assert_contains(answer, [
            "map", "filter", "reduce", "push", "pop", "foreach",
            "find", "includes", "slice", "splice", "concat", "entries", "array"
        ])

    def test_dom_manipulation(self, rag):
        answer, _ = ask(rag, "How do you manipulate the DOM in JavaScript?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["dom", "document", "element", "queryselector", "getelementbyid"])

    def test_functions(self, rag):
        answer, _ = ask(rag, "How do you define a function in JavaScript?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["function", "arrow", "=>", "return", "const"])

    def test_objects(self, rag):
        answer, _ = ask(rag, "How do objects work in JavaScript?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["object", "property", "key", "value", "method", "{"])

    def test_typeof(self, rag):
        answer, _ = ask(rag, "What does the typeof operator do in JavaScript?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["typeof", "type", "string", "number", "operator"])


# ══════════════════════════════════════════════════════════════
# БЛОК 5 — JavaScript Advanced
# ══════════════════════════════════════════════════════════════

class TestJavaScriptAdvanced:

    def test_promises(self, rag):
        answer, _ = ask(rag, "What is a Promise in JavaScript?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["promise", "resolve", "reject", "then", "pending"])

    def test_async_await(self, rag):
        answer, _ = ask(rag, "How does async/await work in JavaScript?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["async", "await", "promise", "asynchronous"])

    def test_closures(self, rag):
        answer, _ = ask(rag, "What is a closure in JavaScript?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["closure", "scope", "function", "variable", "outer"])

    def test_event_loop(self, rag):
        answer, _ = ask(rag, "What is the event loop in JavaScript?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["event loop", "call stack", "queue", "asynchronous", "callback"])

    def test_prototype(self, rag):
        answer, _ = ask(rag, "What is prototypal inheritance in JavaScript?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["prototype", "inherit", "object", "chain"])

    def test_spread_operator(self, rag):
        answer, _ = ask(rag, "What is the spread operator in JavaScript?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["spread", "...", "array", "object", "copy", "expand"])


# ══════════════════════════════════════════════════════════════
# БЛОК 6 — Off-topic
# ══════════════════════════════════════════════════════════════

class TestOffTopic:

    def test_rejects_cooking(self, rag):
        answer, _ = ask(rag, "What is the best recipe for chocolate cake?")
        assert_off_topic(answer)

    def test_rejects_geography(self, rag):
        answer, _ = ask(rag, "What is the capital of France?")
        assert_off_topic(answer)

    def test_rejects_history(self, rag):
        answer, _ = ask(rag, "When did World War II end?")
        assert_off_topic(answer)

    def test_rejects_medicine(self, rag):
        answer, _ = ask(rag, "What are the symptoms of the flu?")
        assert_off_topic(answer)


# ══════════════════════════════════════════════════════════════
# БЛОК 7 — Крос-документні питання
# ══════════════════════════════════════════════════════════════

class TestCrossDocument:

    def test_functions_both_languages(self, rag):
        answer, _ = ask(rag, "How do you define functions? Explain the syntax.")
        assert_not_off_topic(answer)
        assert_contains(answer, ["function", "def", "return", "=>", "parameter"])

    def test_loops_general(self, rag):
        answer, _ = ask(rag, "How do for loops work?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["for", "loop", "iterate", "range"])

    def test_error_handling_general(self, rag):
        answer, _ = ask(rag, "How do you handle errors in code?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["try", "catch", "except", "error", "exception"])

    def test_oop_concepts(self, rag):
        answer, _ = ask(rag, "What is object-oriented programming?")
        assert_not_off_topic(answer)
        assert_contains(answer, ["class", "object", "inherit", "encapsul", "method", "instance"])


# ══════════════════════════════════════════════════════════════
# БЛОК 8 — Швидкість
# ══════════════════════════════════════════════════════════════

class TestLatency:

    @pytest.mark.parametrize("question", [
        "How do you create a NumPy array?",
        "What is a Promise in JavaScript?",
        "How do you define a class in Python?",
        "What is a closure in JavaScript?",
        "How do you read a file in Python?",
    ])
    def test_response_time(self, rag, question):
        _, elapsed = ask(rag, question)
        assert elapsed < MAX_LATENCY_SEC, (
            f"'{question}' — {elapsed:.1f}с (ліміт: {MAX_LATENCY_SEC}с)"
        )