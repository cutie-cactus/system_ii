import json
import pandas as pd
from typing import Dict, List, Any, Optional

class BookRecommendationSystem:
    def __init__(self, dataset_path: str = None, df: pd.DataFrame = None):
        """
        Инициализация рекомендательной системы
        
        Args:
            dataset_path: путь к файлу с данными
            df: готовый DataFrame с данными
        """
        if df is not None:
            self.df = df
        elif dataset_path:
            self.df = pd.read_csv(dataset_path)
        else:
            self.df = pd.DataFrame()
        
        # История запросов
        self.query_history = []
        self.current_filters = {}
        
    def process_query(self, query_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка JSON запроса и возврат результатов
        
        Args:
            query_json: JSON запрос от парсера
            
        Returns:
            Dict с результатами обработки
        """
        try:
            # Сохраняем запрос в историю
            self.query_history.append(query_json.copy())
            
            question_type = query_json.get("question_type", "")
            
            # Обработка в зависимости от типа вопроса
            if question_type == "step_back":
                return self._handle_step_back(query_json)
            elif question_type == "other":
                return self._handle_other(query_json)
            elif question_type == "recommendation":
                return self._handle_recommendation(query_json)
            elif question_type == "search":
                return self._handle_search(query_json)
            elif question_type == "comparison":
                return self._handle_comparison(query_json)
            elif question_type == "general":
                return self._handle_general(query_json)
            else:
                return self._create_response([], "Неизвестный тип запроса")
                
        except Exception as e:
            return self._create_response([], f"Ошибка обработки запроса: {str(e)}")
    
    def _handle_step_back(self, query_json: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка шага назад по истории"""
        step_back = query_json.get("step_back", "")
        
        if step_back == "1":  # Начать сначала
            self.current_filters = {}
            self.query_history = []
            return self._create_response([], "История очищена. Начните новый поиск.")
        elif step_back == "-1":  # Шаг назад
            if len(self.query_history) > 1:
                self.query_history.pop()  # Удаляем текущий запрос
                prev_query = self.query_history[-1]  # Берем предыдущий
                return self.process_query(prev_query)  # Обрабатываем предыдущий
            else:
                return self._create_response([], "Нет предыдущих запросов")
        else:
            return self._create_response([], "Неверное значение step_back")
    
    def _handle_other(self, query_json: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка общих вопросов"""
        num_question = query_json.get("num_question", "")
        
        responses = {
            "вопрос1": "Привет! Я ваш книжный ассистент. Задайте мне вопрос о книгах.",
            "вопрос2": "Я могу искать книги по названию, автору, жанру и другим параметрам.",
            "вопрос3": "Всегда пожалуйста! Рад, что смог помочь.",
            "вопрос4": "Я помогу вам найти книгу по названию или автору, подобрать книгу по жанру или году.",
            "вопрос5": "До свидания! Возвращайтесь, когда понадобятся новые книжные рекомендации."
        }
        
        message = responses.get(num_question, "Неизвестный тип общего вопроса")
        return self._create_response([], message)
    
    def _handle_recommendation(self, query_json: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка рекомендательных запросов"""
        filters = query_json.get("filter", {})
        feedback = query_json.get("feedback", {})
        
        # Обновляем текущие фильтры с учетом фидбека
        self._update_filters_with_feedback(filters, feedback)
        
        # Применяем фильтры
        filtered_books = self._apply_filters(self.df, self.current_filters)
        
        # Если есть лайки, ищем похожие книги
        likes = feedback.get("likes", [])
        if likes:
            similar_books = self._find_similar_books(likes, filtered_books)
            message = f"На основе ваших предпочтений найдено {len(similar_books)} книг"
            return self._create_response(similar_books, message)
        
        # Если есть дизлайки, исключаем похожие
        dislikes = feedback.get("dislikes", [])
        if dislikes:
            filtered_books = self._exclude_similar_books(dislikes, filtered_books)
        
        message = f"Найдено {len(filtered_books)} книг по вашим критериям"
        return self._create_response(filtered_books.to_dict('records'), message)
    
    def _handle_search(self, query_json: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка поисковых запросов"""
        filters = query_json.get("filter", {})
        
        # Обновляем текущие фильтры
        self.current_filters.update(filters)
        
        # Применяем фильтры
        filtered_books = self._apply_filters(self.df, self.current_filters)
        
        message = f"Найдено {len(filtered_books)} книг по вашему запросу"
        return self._create_response(filtered_books.to_dict('records'), message)
    
    def _handle_comparison(self, query_json: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка запросов на сравнение"""
        compare = query_json.get("compare", {})
        count_books = int(compare.get("count_books", 0))
        
        books_to_compare = []
        for i in range(1, count_books + 1):
            title = compare.get(f"title{i}", "")
            author = compare.get(f"author{i}", "")
            
            if title or author:
                book = self._find_book_by_title_author(title, author)
                if book is not None:
                    books_to_compare.append(book)
        
        if len(books_to_compare) >= 2:
            comparison_result = self._compare_books(books_to_compare)
            message = f"Сравнение {len(books_to_compare)} книг"
            return self._create_response(comparison_result, message)
        else:
            return self._create_response([], "Не найдено достаточного количества книг для сравнения")
    
    def _handle_general(self, query_json: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка общих вопросов о книгах"""
        filters = query_json.get("filter", {})
        
        # Для общих вопросов обычно ищем одну конкретную книгу
        filtered_books = self._apply_filters(self.df, filters)
        
        if len(filtered_books) == 1:
            message = "Информация о книге"
            return self._create_response([filtered_books.iloc[0].to_dict()], message)
        elif len(filtered_books) > 1:
            message = f"Найдено несколько книг ({len(filtered_books)}). Уточните запрос."
            return self._create_response(filtered_books.to_dict('records'), message)
        else:
            return self._create_response([], "Книга не найдена")
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Применение фильтров к DataFrame"""
        filtered_df = df.copy()
        
        # Фильтрация по автору (массив)
        authors = filters.get("author", [])
        if authors:
            author_filter = filtered_df['author'].str.lower().isin([a.lower() for a in authors])
            filtered_df = filtered_df[author_filter]
        
        # Фильтрация по издателю (массив)
        publishers = filters.get("publisher", [])
        if publishers:
            publisher_filter = filtered_df['publisher'].str.lower().isin([p.lower() for p in publishers])
            filtered_df = filtered_df[publisher_filter]
        
        # Фильтрация по году
        year_from = filters.get("year_from")
        year_to = filters.get("year_to")
        if year_from:
            filtered_df = filtered_df[filtered_df['year'] >= int(year_from)]
        if year_to:
            filtered_df = filtered_df[filtered_df['year'] <= int(year_to)]
        
        # Фильтрация по языку (массив)
        languages = filters.get("language", [])
        if languages:
            language_filter = filtered_df['language'].str.lower().isin([l.lower() for l in languages])
            filtered_df = filtered_df[language_filter]
        
        # Фильтрация по возрастному ограничению (массив)
        age_restrictions = filters.get("age_restriction", [])
        if age_restrictions:
            age_filter = filtered_df['age_restriction'].isin(age_restrictions)
            filtered_df = filtered_df[age_filter]
        
        # Фильтрация по жанру (массив)
        genres = filters.get("genre", [])
        if genres:
            # Для жанров используем частичное совпадение, так как книга может иметь несколько жанров
            genre_pattern = '|'.join(genres)
            genre_filter = filtered_df['genre'].str.contains(genre_pattern, case=False, na=False)
            filtered_df = filtered_df[genre_filter]
        
        # Фильтрация по количеству страниц
        pages_from = filters.get("pages_from")
        pages_to = filters.get("pages_to")
        if pages_from:
            filtered_df = filtered_df[filtered_df['pages'] >= int(pages_from)]
        if pages_to:
            filtered_df = filtered_df[filtered_df['pages'] <= int(pages_to)]
        
        # Фильтрация по наличию иллюстраций
        has_illustrations = filters.get("has_illustrations")
        if has_illustrations:
            if has_illustrations.lower() == "true":
                filtered_df = filtered_df[filtered_df['has_illustrations'] == True]
            elif has_illustrations.lower() == "false":
                filtered_df = filtered_df[filtered_df['has_illustrations'] == False]
        
        return filtered_df
    
    def _update_filters_with_feedback(self, new_filters: Dict[str, Any], feedback: Dict[str, List[str]]):
        """Обновление текущих фильтров с учетом обратной связи"""
        # Обновляем фильтры новыми значениями
        for key, value in new_filters.items():
            if value:  # Если значение не пустое
                if isinstance(value, list) and key in self.current_filters:
                    # Для массивов объединяем значения
                    self.current_filters[key] = list(set(self.current_filters[key] + value))
                else:
                    self.current_filters[key] = value
        
        # Учитываем feedback в фильтрах
        likes = feedback.get("likes", [])
        dislikes = feedback.get("dislikes", [])
        
        # Для лайков можно добавить в приоритетные жанры/авторов
        if likes:
            for book_title in likes:
                book = self._find_book_by_title(book_title)
                if book is not None:
                    # Добавляем автора и жанр в приоритеты
                    if 'author' not in self.current_filters:
                        self.current_filters['author'] = []
                    if book['author'] not in self.current_filters['author']:
                        self.current_filters['author'].append(book['author'])
                    
                    if 'genre' not in self.current_filters:
                        self.current_filters['genre'] = []
                    # Жанр может быть строкой с несколькими жанрами
                    genres = book['genre'].split(',') if isinstance(book['genre'], str) else [book['genre']]
                    for genre in genres:
                        genre = genre.strip()
                        if genre not in self.current_filters['genre']:
                            self.current_filters['genre'].append(genre)
    
    def _find_similar_books(self, liked_titles: List[str], base_books: pd.DataFrame) -> List[Dict[str, Any]]:
        """Поиск книг, похожих на понравившиеся"""
        similar_books = []
        
        for title in liked_titles:
            book = self._find_book_by_title(title)
            if book is not None:
                # Ищем книги того же автора
                author_books = base_books[base_books['author'] == book['author']]
                similar_books.extend(author_books.to_dict('records'))
                
                # Ищем книги того же жанра
                if isinstance(book['genre'], str):
                    genres = [g.strip() for g in book['genre'].split(',')]
                    for genre in genres:
                        genre_books = base_books[base_books['genre'].str.contains(genre, case=False, na=False)]
                        similar_books.extend(genre_books.to_dict('records'))
        
        # Удаляем дубликаты и уже понравившиеся книги
        seen_ids = set()
        unique_books = []
        
        for book in similar_books:
            if book['id'] not in seen_ids and book['title'] not in liked_titles:
                seen_ids.add(book['id'])
                unique_books.append(book)
        
        return unique_books[:10]  # Ограничиваем количество рекомендаций
    
    def _exclude_similar_books(self, disliked_titles: List[str], base_books: pd.DataFrame) -> pd.DataFrame:
        """Исключение книг, похожих на непонравившиеся"""
        filtered_books = base_books.copy()
        
        for title in disliked_titles:
            book = self._find_book_by_title(title)
            if book is not None:
                # Исключаем книги того же автора
                filtered_books = filtered_books[filtered_books['author'] != book['author']]
        
        return filtered_books
    
    def _find_book_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Поиск книги по названию"""
        if self.df.empty:
            return None
        
        book = self.df[self.df['title'].str.lower() == title.lower()]
        if not book.empty:
            return book.iloc[0].to_dict()
        return None
    
    def _find_book_by_title_author(self, title: str, author: str) -> Optional[Dict[str, Any]]:
        """Поиск книги по названию и автору"""
        if self.df.empty:
            return None
        
        if title:
            book = self.df[self.df['title'].str.lower() == title.lower()]
        elif author:
            book = self.df[self.df['author'].str.lower() == author.lower()]
        else:
            return None
        
        if not book.empty:
            return book.iloc[0].to_dict()
        return None
    
    def _compare_books(self, books: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Сравнение книг по различным параметрам"""
        comparison = []
        
        for book in books:
            comparison.append({
                'id': book.get('id', ''),
                'title': book.get('title', ''),
                'author': book.get('author', ''),
                'year': book.get('year', ''),
                'genre': book.get('genre', ''),
                'pages': book.get('pages', ''),
                'language': book.get('language', ''),
                'age_restriction': book.get('age_restriction', ''),
                'publisher': book.get('publisher', ''),
                'has_illustrations': book.get('has_illustrations', False)
            })
        
        return comparison
    
    def _create_response(self, books: List[Dict[str, Any]], message: str) -> Dict[str, Any]:
        """Создание стандартного ответа"""
        return {
            "success": True,
            "message": message,
            "books": books,
            "total_count": len(books),
            "current_filters": self.current_filters
        }


# Пример использования
def main():
    # Создаем тестовый dataset
    test_data = {
        'id': [1, 2, 3, 4, 5],
        'title': ['Гарри Поттер и философский камень', '1984', 'Война и мир', 'Мастер и Маргарита', 'Анна Каренина'],
        'author': ['Джоан Роулинг', 'Джордж Оруэлл', 'Лев Толстой', 'Михаил Булгаков', 'Лев Толстой'],
        'publisher': ['Росмэн', 'АСТ', 'Эксмо', 'Азбука', 'Эксмо'],
        'year': [2001, 1949, 1869, 1967, 1877],
        'language': ['русский', 'русский', 'русский', 'русский', 'русский'],
        'age_restriction': ['12+', '16+', '12+', '16+', '12+'],
        'genre': ['фэнтези', 'антиутопия', 'роман', 'роман', 'роман'],
        'pages': [400, 328, 1225, 480, 864],
        'has_illustrations': [True, False, False, False, False]
    }
    
    df = pd.DataFrame(test_data)
    
    # Инициализируем систему
    recommender = BookRecommendationSystem(df=df)
    
    # Тестовые запросы
    test_queries = [
        {
            "question_type": "search",
            "filter": {
                "author": [],
                "publisher": [],
                "year_from": "",
                "year_to": "",
                "language": [],
                "age_restriction": [],
                "genre": ["фэнтези"],
                "pages_from": "",
                "pages_to": "",
                "has_illustrations": ""
            },
            "compare": {
                "count_books": "",
                "title1": "",
                "author1": "",
                "title2": "",
                "author2": ""
            },
            "feedback": {
                "likes": [],
                "dislikes": []
            },
            "num_question": "",
            "step_back": ""
        },
        {
            "question_type": "recommendation",
            "filter": {
                "author": [],
                "publisher": [],
                "year_from": "",
                "year_to": "",
                "language": [],
                "age_restriction": [],
                "genre": [],
                "pages_from": "",
                "pages_to": "",
                "has_illustrations": ""
            },
            "compare": {
                "count_books": "",
                "title1": "",
                "author1": "",
                "title2": "",
                "author2": ""
            },
            "feedback": {
                "likes": ["Гарри Поттер и философский камень"],
                "dislikes": []
            },
            "num_question": "",
            "step_back": ""
        }
    ]
    
    # Обрабатываем запросы
    for i, query in enumerate(test_queries):
        print(f"Запрос {i + 1}:")
        result = recommender.process_query(query)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("-" * 50)

if __name__ == "__main__":
    main()
