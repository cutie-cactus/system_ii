"""
Обработка распарсенных запросов
"""
import pandas as pd
from typing import Dict, Any, List, Tuple
from data_loader import BookDataLoader
from book_metrics import BookDistanceMetrics

class QueryProcessor:
    def __init__(self, data_loader: BookDataLoader):
        self.data_loader = data_loader
        self.metrics_full = None
        self.metrics_filtered = None
        
    def initialize_metrics(self):
        """Инициализация метрик для данных"""
        if self.data_loader.df is not None:
            self.metrics_full = BookDistanceMetrics(self.data_loader.df)
            if self.data_loader.filtered_df is not None:
                self.metrics_filtered = BookDistanceMetrics(self.data_loader.filtered_df)
    
    def process_query(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка распарсенного запроса
        
        Returns:
            Словарь с результатами обработки:
            {
                'query_type': тип запроса,
                'filtered_books': отфильтрованные книги (DataFrame),
                'liked_indices': индексы понравившихся книг,
                'disliked_indices': индексы непонравившихся книг,
                'comparison_books': книги для сравнения,
                'message': сообщение о результате
            }
        """
        result = {
            'query_type': parsed_query.get('question_type', ''),
            'filtered_books': None,
            'liked_indices': [],
            'disliked_indices': [],
            'comparison_books': [],
            'message': ''
        }
        
        # Получаем тип запроса
        query_type = parsed_query.get('question_type', '')
        
        # Обработка фильтров
        if query_type in ['search', 'general', 'recommendation']:
            filter_result = self._apply_filters(parsed_query.get('filter', {}))
            result['filtered_books'] = filter_result['filtered_books']
            result['message'] = filter_result['message']
            
            # Обновляем метрики для отфильтрованных данных
            if result['filtered_books'] is not None and len(result['filtered_books']) > 0:
                self.metrics_filtered = BookDistanceMetrics(result['filtered_books'])
        
        # Обработка лайков/дизлайков
        if query_type == 'recommendation':
            feedback = parsed_query.get('feedback', {})
            likes = feedback.get('likes', [])
            dislikes = feedback.get('dislikes', [])
            
            # Получаем индексы лайков/дизлайков
            result['liked_indices'] = self.data_loader.get_book_indices_by_titles(likes)
            result['disliked_indices'] = self.data_loader.get_book_indices_by_titles(dislikes)
            
            if not result['liked_indices']:
                result['message'] += "\n⚠️ Не найдены книги из списка 'понравилось'"
        
        # Обработка сравнения
        elif query_type == 'comparison':
            compare = parsed_query.get('compare', {})
            title1 = compare.get('title1', '')
            author1 = compare.get('author1', '')
            title2 = compare.get('title2', '')
            author2 = compare.get('author2', '')
            
            book1 = self.data_loader.get_book_by_title_author(title1, author1)
            book2 = self.data_loader.get_book_by_title_author(title2, author2)
            
            if book1 is not None:
                result['comparison_books'].append(book1)
            if book2 is not None:
                result['comparison_books'].append(book2)
            
            if len(result['comparison_books']) < 2:
                result['message'] += "\n⚠️ Не найдены книги для сравнения"
        
        return result
    
    def _apply_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Применение фильтров к данным"""
        result = {
            'filtered_books': None,
            'message': ''
        }
        
        # Подготавливаем критерии фильтрации
        filter_criteria = {}
        
        # Жанры
        if 'genre' in filters and filters['genre']:
            filter_criteria['genre'] = filters['genre']
        
        # Авторы
        if 'author' in filters and filters['author']:
            filter_criteria['author'] = filters['author']
        
        # Год издания
        if 'year_from' in filters and filters['year_from']:
            try:
                filter_criteria['year_from'] = int(filters['year_from'])
            except ValueError:
                pass
        if 'year_to' in filters and filters['year_to']:
            try:
                filter_criteria['year_to'] = int(filters['year_to'])
            except ValueError:
                pass
        
        # Страницы
        if 'pages_from' in filters and filters['pages_from']:
            try:
                filter_criteria['pages_from'] = int(filters['pages_from'])
            except ValueError:
                pass
        if 'pages_to' in filters and filters['pages_to']:
            try:
                filter_criteria['pages_to'] = int(filters['pages_to'])
            except ValueError:
                pass
        
        # Язык
        if 'language' in filters and filters['language']:
            filter_criteria['language'] = filters['language']
        
        # Иллюстрации
        if 'has_illustrations' in filters and filters['has_illustrations']:
            ill_value = filters['has_illustrations']
            if ill_value.lower() in ['есть', 'да', 'true', '1']:
                filter_criteria['has_illustrations'] = True
            elif ill_value.lower() in ['нет', 'нету', 'false', '0']:
                filter_criteria['has_illustrations'] = False
        
        # Применяем фильтры
        try:
            filtered_books = self.data_loader.filter_books(filter_criteria)
            result['filtered_books'] = filtered_books
            result['message'] = f"Найдено {len(filtered_books)} книг"
        except Exception as e:
            result['message'] = f"Ошибка при фильтрации: {str(e)}"
        
        return result
    
    def get_book_info(self, book_idx: int, is_filtered: bool = False) -> Dict[str, Any]:
        """Получение информации о книге"""
        df = self.data_loader.filtered_df if is_filtered else self.data_loader.df
        
        if df is None or book_idx >= len(df):
            return None
        
        book = df.iloc[book_idx]
        return {
            'title': book['title'],
            'author': book['author'],
            'genre': book['genre'],
            'year': book['year'],
            'pages': book['pages'],
            'publisher': book['publisher'],
            'language': book['language'],
            'age_restriction': book['age_restriction'],
            'has_illustrations': 'Есть' if book['has_illustrations'] == 1 else 'Нет'
        }