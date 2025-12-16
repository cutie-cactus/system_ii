"""
Загрузка и подготовка данных
"""
import pandas as pd
import numpy as np
from typing import Optional

class BookDataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.filtered_df = None
        
    def load_data(self) -> pd.DataFrame:
        """Загрузка данных из CSV файла"""
        try:
            self.df = pd.read_csv(self.data_path)
            self.filtered_df = self.df.copy()
            print(f"✅ Данные загружены: {len(self.df)} книг")
            return self.df
        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            raise
            
    def filter_books(self, filter_criteria: dict) -> pd.DataFrame:
        """
        Фильтрация книг по заданным критериям
        
        Args:
            filter_criteria: словарь с критериями фильтрации
                Пример:
                {
                    'genre': ['фэнтези', 'фантастика'],
                    'author': ['Дж. Роулинг', 'Дж. Толкин'],
                    'year_from': 2000,
                    'year_to': 2020,
                    'pages_from': 100,
                    'pages_to': 500,
                    'language': ['русский'],
                    'has_illustrations': True
                }
        """
        if self.df is None:
            raise ValueError("Данные не загружены")
            
        filtered = self.df.copy()
        
        # Фильтр по жанру
        if 'genre' in filter_criteria and filter_criteria['genre']:
            genres = [g for g in filter_criteria['genre'] if g]
            if genres:
                filtered = filtered[filtered['genre'].isin(genres)]
        
        # Фильтр по автору
        if 'author' in filter_criteria and filter_criteria['author']:
            authors = [a for a in filter_criteria['author'] if a]
            if authors:
                filtered = filtered[filtered['author'].isin(authors)]
        
        # Фильтр по году
        if 'year_from' in filter_criteria and filter_criteria['year_from']:
            filtered = filtered[filtered['year'] >= filter_criteria['year_from']]
        if 'year_to' in filter_criteria and filter_criteria['year_to']:
            filtered = filtered[filtered['year'] <= filter_criteria['year_to']]
        
        # Фильтр по страницам
        if 'pages_from' in filter_criteria and filter_criteria['pages_from']:
            filtered = filtered[filtered['pages'] >= filter_criteria['pages_from']]
        if 'pages_to' in filter_criteria and filter_criteria['pages_to']:
            filtered = filtered[filtered['pages'] <= filter_criteria['pages_to']]
        
        # Фильтр по языку
        if 'language' in filter_criteria and filter_criteria['language']:
            languages = [l for l in filter_criteria['language'] if l]
            if languages:
                filtered = filtered[filtered['language'].isin(languages)]
        
        # Фильтр по иллюстрациям
        if 'has_illustrations' in filter_criteria:
            has_ill = filter_criteria['has_illustrations']
            if has_ill == "Есть" or has_ill is True:
                filtered = filtered[filtered['has_illustrations'] == 1]
            elif has_ill == "Нет" or has_ill is False:
                filtered = filtered[filtered['has_illustrations'] == 0]
        
        self.filtered_df = filtered.reset_index(drop=True)
        print(f"✅ Отфильтровано: {len(self.filtered_df)} книг")
        return self.filtered_df
    
    def reset_filters(self):
        """Сброс фильтров"""
        self.filtered_df = self.df.copy()
        return self.filtered_df
    
    def get_book_by_title_author(self, title: str, author: str = None) -> Optional[pd.Series]:
        """Поиск книги по названию и автору"""
        if author:
            mask = (self.df['title'].str.contains(title, case=False, na=False)) & \
                   (self.df['author'].str.contains(author, case=False, na=False))
        else:
            mask = self.df['title'].str.contains(title, case=False, na=False)
        
        results = self.df[mask]
        if len(results) > 0:
            return results.iloc[0]
        return None
    
    def get_book_indices_by_titles(self, titles: list, authors: list = None) -> list:
        """Получение индексов книг по названиям"""
        indices = []
        for i, title in enumerate(titles):
            author = authors[i] if authors and i < len(authors) else None
            book = self.get_book_by_title_author(title, author)
            if book is not None:
                indices.append(book.name)
        return indices