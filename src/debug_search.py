from src.document_loader import load_and_split_documents

# Встав точну назву свого файлу!
file_path = "Oform-24-25.pdf"

print(f"🔍 Сканую файл {file_path}...")
docs = load_and_split_documents(file_path)

found = False
print("\n--- РЕЗУЛЬТАТИ ПОШУКУ СЛОВА 'Times' ---")
for i, doc in enumerate(docs):
    # Шукаємо "Times" або "шрифт" (без залежності від регістру)
    if "times" in doc.page_content.lower() or "шрифт" in doc.page_content.lower():
        print(f"\n✅ ЗНАЙДЕНО у фрагменті #{i}:")
        print(f"...{doc.page_content[:300]}...") # Друкуємо початок знайденого шматка
        found = True

if not found:
    print("\n❌ СЛОВА 'Times' або 'шрифт' НЕ ЗНАЙДЕНО у тексті!")
    print("Це означає, що pdfplumber не зміг прочитати цей конкретний абзац.")