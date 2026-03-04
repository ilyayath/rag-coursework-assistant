# conftest.py — кореневий файл, додає src/ до PYTHONPATH
import sys
import os

# Додаємо корінь проєкту до sys.path щоб pytest бачив пакет src
sys.path.insert(0, os.path.dirname(__file__))