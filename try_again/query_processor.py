"""
Обработка распарсенных запросов с поддержкой истории и состояния
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from data_loader import BookDataLoader
from book_metrics import BookDistanceMetrics
from collections import deque

class QueryState:
    """Класс для хранения состояния запросов с историей"""
    
    MAX_HISTORY_STEPS = 5  # Максимальная глубина истории
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Полный сброс состояния"""
        self.current_state = {
            'filter': {
                'author': [],
                'publisher': [],
                'year_from': '',
                'year_to': '',
                'language': [],
                'age_restriction': [],
                'genre': [],
                'pages_from': '',
                'pages_to': '',
                'has_illustrations': ''
            },
            'feedback': {
                'likes': [],      # Названия книг, которые нравятся
                'dislikes': []    # Названия книг, которые не нравятся
            },
            'filtered_books': None,  # DataFrame с отфильтрованными книгами
            'liked_indices': [],     # Индексы лайков в полном датасете
            'disliked_indices': []   # Индексы дизлайков в полном датасете
        }
        
        self.history = deque(maxlen=self.MAX_HISTORY_STEPS)
        self.history.append(self._copy_state(self.current_state))
    
    def update(self, new_filters: Dict[str, Any], new_feedback: Dict[str, List[str]],
               filtered_books: pd.DataFrame, liked_indices: List[int], 
               disliked_indices: List[int]) -> Dict[str, Any]:
        """
        Обновление состояния с сохранением в историю
        
        Args:
            new_filters: новые фильтры (частичные)
            new_feedback: новые лайки/дизлайки (частичные)
            filtered_books: отфильтрованный DataFrame
            liked_indices: индексы лайков в полном датасете
            disliked_indices: индексы дизлайков в полном датасете
        
        Returns:
            Объединенное текущее состояние
        """
        # Сохраняем текущее состояние в историю
        self.history.append(self._copy_state(self.current_state))
        
        # Объединяем фильтры
        for key, value in new_filters.items():
            if key in self.current_state['filter']:
                if isinstance(value, list) and value:
                    if key in ['author', 'publisher', 'language', 'age_restriction', 'genre']:
                        # Для списков - добавляем новые элементы
                        current = self.current_state['filter'][key]
                        if isinstance(current, list):
                            current.extend([v for v in value if v not in current])
                        else:
                            self.current_state['filter'][key] = value.copy()
                    else:
                        # Для одиночных значений - заменяем
                        self.current_state['filter'][key] = value
        
        # Объединяем feedback (лайки/дизлайки)
        for key in ['likes', 'dislikes']:
            if key in new_feedback and new_feedback[key]:
                current_list = self.current_state['feedback'][key]
                for item in new_feedback[key]:
                    if item not in current_list:
                        current_list.append(item)
        
        # Обновляем остальные поля
        self.current_state['filtered_books'] = filtered_books
        self.current_state['liked_indices'] = liked_indices
        self.current_state['disliked_indices'] = disliked_indices
        
        return self.current_state
    
    def step_back(self) -> Optional[Dict[str, Any]]:
        """
        Возврат на шаг назад
        
        Returns:
            Предыдущее состояние или None, если история пуста
        """
        if len(self.history) > 1:
            # Удаляем текущее состояние
            self.history.pop()
            # Восстанавливаем предыдущее
            self.current_state = self._copy_state(self.history[-1])
            return self.current_state
        return None
    
    def get_current_state(self) -> Dict[str, Any]:
        """Получение копии текущего состояния"""
        return self._copy_state(self.current_state)
    
    def _copy_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Создание глубокой копии состояния"""
        return {
            'filter': {
                key: (value.copy() if isinstance(value, list) else value)
                for key, value in state['filter'].items()
            },
            'feedback': {
                'likes': state['feedback']['likes'].copy(),
                'dislikes': state['feedback']['dislikes'].copy()
            },
            'filtered_books': state['filtered_books'].copy() if state['filtered_books'] is not None else None,
            'liked_indices': state['liked_indices'].copy(),
            'disliked_indices': state['disliked_indices'].copy()
        }
    
    def get_history_info(self) -> Dict[str, Any]:
        """Информация об истории"""
        return {
            'current_step': len(self.history),
            'max_steps': self.MAX_HISTORY_STEPS,
            'can_go_back': len(self.history) > 1
        }


class QueryProcessor:
    """
    Обработка распарсенных запросов с поддержкой истории
    """
    def __init__(self, data_loader: BookDataLoader):
        self.data_loader = data_loader
        self.state = QueryState()
        self.metrics_full = None
        self.metrics_filtered = None
        
    def initialize_metrics(self):
        """Инициализация метрик для данных"""
        if self.data_loader.df is not None:
            self.metrics_full = BookDistanceMetrics(self.data_loader.df)
            # Инициализируем filtered_books как полный датасет
            self.state.current_state['filtered_books'] = self.data_loader.df.copy()
            self.metrics_filtered = BookDistanceMetrics(self.state.current_state['filtered_books'])
    
    def process_query(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка распарсенного запроса с учетом истории
        
        Returns:
            Словарь с результатами обработки
        """
        result = {
            'query_type': parsed_query.get('question_type', ''),
            'new_filters': {},
            'new_feedback': {'likes': [], 'dislikes': []},
            'filtered_books': None,
            'liked_indices': [],
            'disliked_indices': [],
            'comparison_books': [],
            'message': '',
            'history_info': self.state.get_history_info()
        }
        
        query_type = parsed_query.get('question_type', '')
        
        # Обработка специальных команд
        if query_type == 'step_back':
            return self._handle_step_back()
        elif query_type == 'other' and parsed_query.get('num_question') == 'заново':
            return self._handle_reset()
        
        # Получаем текущее состояние
        current_state = self.state.get_current_state()
        
        # Для новых запросов применяем наслоение
        if query_type in ['search', 'general', 'recommendation']:
            # Извлекаем новые фильтры из запроса
            new_filters = self._extract_filters(parsed_query.get('filter', {}))
            result['new_filters'] = new_filters
            
            # Объединяем с текущими фильтрами
            combined_filters = self._combine_filters(current_state['filter'], new_filters)
            
            # Применяем объединенные фильтры
            filter_result = self._apply_combined_filters(combined_filters)
            result['filtered_books'] = filter_result['filtered_books']
            result['message'] = filter_result['message']
            
            # Обновляем метрики для отфильтрованных данных
            if result['filtered_books'] is not None and len(result['filtered_books']) > 0:
                self.metrics_filtered = BookDistanceMetrics(result['filtered_books'])
        
        # Обработка лайков/дизлайков
        if query_type == 'recommendation' or 'feedback' in parsed_query:
            feedback = parsed_query.get('feedback', {})
            new_likes = feedback.get('likes', [])
            new_dislikes = feedback.get('dislikes', [])
            
            result['new_feedback']['likes'] = new_likes
            result['new_feedback']['dislikes'] = new_dislikes
            
            # Получаем индексы новых лайков/дизлайков
            new_liked_indices = self.data_loader.get_book_indices_by_titles(new_likes)
            new_disliked_indices = self.data_loader.get_book_indices_by_titles(new_dislikes)
            
            # Объединяем с текущими
            combined_liked_indices = list(set(current_state['liked_indices'] + new_liked_indices))
            combined_disliked_indices = list(set(current_state['disliked_indices'] + new_disliked_indices))
            
            result['liked_indices'] = combined_liked_indices
            result['disliked_indices'] = combined_disliked_indices
            
            if new_likes and not new_liked_indices:
                result['message'] += "\n⚠️ Не найдены книги из списка 'понравилось'"
            if new_dislikes and not new_disliked_indices:
                result['message'] += "\n⚠️ Не найдены книги из списка 'не понравилось'"
        
        # Обновляем состояние
        if query_type in ['search', 'general', 'recommendation']:
            updated_state = self.state.update(
                new_filters=result['new_filters'],
                new_feedback=result['new_feedback'],
                filtered_books=result['filtered_books'],
                liked_indices=result['liked_indices'],
                disliked_indices=result['disliked_indices']
            )
            
            # Обновляем результат с актуальными данными из состояния
            result['filtered_books'] = updated_state['filtered_books']
            result['liked_indices'] = updated_state['liked_indices']
            result['disliked_indices'] = updated_state['disliked_indices']
        
        # Обработка сравнения (не влияет на состояние)
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
    
    def _handle_step_back(self) -> Dict[str, Any]:
        """Обработка команды 'назад'"""
        previous_state = self.state.step_back()
        
        if previous_state is None:
            return {
                'query_type': 'step_back',
                'message': '❌ Невозможно вернуться назад - история пуста',
                'history_info': self.state.get_history_info()
            }
        
        # Обновляем метрики для восстановленного состояния
        if previous_state['filtered_books'] is not None:
            self.metrics_filtered = BookDistanceMetrics(previous_state['filtered_books'])
        
        return {
            'query_type': 'step_back',
            'filtered_books': previous_state['filtered_books'],
            'liked_indices': previous_state['liked_indices'],
            'disliked_indices': previous_state['disliked_indices'],
            'message': '↩️  Возврат на шаг назад',
            'history_info': self.state.get_history_info()
        }
    
    def _handle_reset(self) -> Dict[str, Any]:
        """Обработка команды 'заново'"""
        self.state.reset()
        
        # Сбрасываем фильтры в data_loader
        self.data_loader.reset_filters()
        
        # Инициализируем метрики заново
        self.initialize_metrics()
        
        return {
            'query_type': 'reset',
            'filtered_books': self.data_loader.df.copy(),
            'liked_indices': [],
            'disliked_indices': [],
            'message': '🔄 Начинаем заново. Все фильтры и предпочтения сброшены.',
            'history_info': self.state.get_history_info()
        }
    
    def _extract_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Извлечение фильтров из распарсенного запроса"""
        extracted = {}
        
        # Жанры
        if 'genre' in filters and filters['genre']:
            extracted['genre'] = filters['genre']
        
        # Авторы
        if 'author' in filters and filters['author']:
            extracted['author'] = filters['author']
        
        # Год издания
        if 'year_from' in filters and filters['year_from']:
            try:
                extracted['year_from'] = int(filters['year_from'])
            except ValueError:
                pass
        if 'year_to' in filters and filters['year_to']:
            try:
                extracted['year_to'] = int(filters['year_to'])
            except ValueError:
                pass
        
        # Страницы
        if 'pages_from' in filters and filters['pages_from']:
            try:
                extracted['pages_from'] = int(filters['pages_from'])
            except ValueError:
                pass
        if 'pages_to' in filters and filters['pages_to']:
            try:
                extracted['pages_to'] = int(filters['pages_to'])
            except ValueError:
                pass
        
        # Язык
        if 'language' in filters and filters['language']:
            extracted['language'] = filters['language']
        
        # Иллюстрации
        if 'has_illustrations' in filters and filters['has_illustrations']:
            ill_value = filters['has_illustrations']
            if ill_value.lower() in ['есть', 'да', 'true', '1']:
                extracted['has_illustrations'] = True
            elif ill_value.lower() in ['нет', 'нету', 'false', '0']:
                extracted['has_illustrations'] = False
        
        return extracted
    
    def _combine_filters(self, current: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """Объединение текущих и новых фильтров"""
        combined = current.copy()
        
        for key, value in new.items():
            if key in combined:
                if isinstance(value, list) and value:
                    # Для списков - добавляем новые элементы
                    if isinstance(combined[key], list):
                        combined[key].extend([v for v in value if v not in combined[key]])
                    else:
                        combined[key] = value.copy()
                elif value:  # Для одиночных значений - заменяем
                    combined[key] = value
        
        return combined
    
    def _apply_combined_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Применение объединенных фильтров к данным"""
        result = {
            'filtered_books': None,
            'message': ''
        }
        
        # Подготавливаем критерии фильтрации для data_loader
        filter_criteria = {}
        
        # Преобразуем формат фильтров для data_loader
        if 'genre' in filters and filters['genre']:
            filter_criteria['genre'] = filters['genre']
        if 'author' in filters and filters['author']:
            filter_criteria['author'] = filters['author']
        if 'year_from' in filters and filters['year_from']:
            filter_criteria['year_from'] = filters['year_from']
        if 'year_to' in filters and filters['year_to']:
            filter_criteria['year_to'] = filters['year_to']
        if 'pages_from' in filters and filters['pages_from']:
            filter_criteria['pages_from'] = filters['pages_from']
        if 'pages_to' in filters and filters['pages_to']:
            filter_criteria['pages_to'] = filters['pages_to']
        if 'language' in filters and filters['language']:
            filter_criteria['language'] = filters['language']
        if 'has_illustrations' in filters:
            filter_criteria['has_illustrations'] = filters['has_illustrations']
        
        # Применяем фильтры
        try:
            filtered_books = self.data_loader.filter_books(filter_criteria)
            result['filtered_books'] = filtered_books
            
            # Формируем информативное сообщение
            filter_info = []
            if 'genre' in filter_criteria:
                filter_info.append(f"жанры: {', '.join(filter_criteria['genre'])}")
            if 'author' in filter_criteria:
                filter_info.append(f"авторы: {', '.join(filter_criteria['author'])}")
            if 'year_from' in filter_criteria or 'year_to' in filter_criteria:
                year_from = filter_criteria.get('year_from', 'любой')
                year_to = filter_criteria.get('year_to', 'любой')
                filter_info.append(f"год: {year_from}-{year_to}")
            
            if filter_info:
                result['message'] = f"Применены фильтры: {'; '.join(filter_info)}. "
            result['message'] += f"Найдено {len(filtered_books)} книг"
            
        except Exception as e:
            result['message'] = f"Ошибка при фильтрации: {str(e)}"
        
        return result
    
    def get_current_state_info(self) -> Dict[str, Any]:
        """Получение информации о текущем состоянии"""
        state = self.state.get_current_state()
        
        info = {
            'active_filters': {},
            'preferences': {
                'likes_count': len(state['feedback']['likes']),
                'dislikes_count': len(state['feedback']['dislikes']),
                'likes': state['feedback']['likes'][:5],  # Первые 5 для показа
                'dislikes': state['feedback']['dislikes'][:5]
            },
            'books_count': len(state['filtered_books']) if state['filtered_books'] is not None else 0,
            'history': self.state.get_history_info()
        }
        
        # Собираем активные фильтры
        for key, value in state['filter'].items():
            if value:
                if isinstance(value, list) and value:
                    info['active_filters'][key] = value
                elif value not in ['', 0, False]:
                    info['active_filters'][key] = value
        
        return info
    
    def get_book_info(self, book_idx: int, is_filtered: bool = False) -> Dict[str, Any]:
        """Получение информации о книге"""
        if is_filtered and self.state.current_state['filtered_books'] is not None:
            df = self.state.current_state['filtered_books']
        else:
            df = self.data_loader.df
        
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