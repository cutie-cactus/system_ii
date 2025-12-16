"""
Вспомогательные функции
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json

def save_results_to_file(results: Dict[str, Any], filename: str = "results.json"):
    """Сохранение результатов в файл"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # Преобразуем DataFrame в словарь для сериализации
            if 'data' in results and isinstance(results['data'], pd.DataFrame):
                results['data'] = results['data'].to_dict(orient='records')
            
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"✅ Результаты сохранены в {filename}")
    except Exception as e:
        print(f"❌ Ошибка сохранения результатов: {e}")

def load_results_from_file(filename: str = "results.json") -> Dict[str, Any]:
    """Загрузка результатов из файла"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Ошибка загрузки результатов: {e}")
        return {}

def print_book_info(book: pd.Series):
    """Красивый вывод информации о книге"""
    if book is None:
        print("❌ Книга не найдена")
        return
    
    print("📚" + "=" * 60)
    print(f"Название: {book['title']}")
    print(f"Автор: {book['author']}")
    print(f"Жанр: {book['genre']}")
    print(f"Год издания: {book['year']}")
    print(f"Количество страниц: {book['pages']}")
    print(f"Издательство: {book['publisher']}")
    print(f"Язык: {book['language']}")
    print(f"Возрастное ограничение: {book['age_restriction']}+")
    print(f"Иллюстрации: {'Есть' if book['has_illustrations'] == 1 else 'Нет'}")
    print("=" * 60)

def validate_filter_criteria(criteria: Dict[str, Any]) -> bool:
    """Проверка корректности критериев фильтрации"""
    try:
        if 'year_from' in criteria and criteria['year_from']:
            int(criteria['year_from'])
        if 'year_to' in criteria and criteria['year_to']:
            int(criteria['year_to'])
        if 'pages_from' in criteria and criteria['pages_from']:
            int(criteria['pages_from'])
        if 'pages_to' in criteria and criteria['pages_to']:
            int(criteria['pages_to'])
        return True
    except ValueError:
        return False