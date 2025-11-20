import json
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import deque

class BookRecommendationSystem:
    def __init__(self, dataset_path: str = None, df: pd.DataFrame = None):
        """
        Инициализация рекомендательной системы
        """
        if df is not None:
            self.df = df
        elif dataset_path:
            self.df = pd.read_csv(dataset_path)
        else:
            # Создаем тестовый dataset для демонстрации
            self.df = self._create_sample_data()
        
        # История запросов (последние 10)
        self.query_history = deque(maxlen=10)
        self.current_filters = {}
        self.last_results = []
        
    def _create_sample_data(self):
        """Загрузка данных из CSV файла"""
        try:
            # Загружаем данные из CSV файла
            df = pd.read_csv('../DataBooks.csv')
            print(f"✅ Данные загружены: {len(df)} книг")
            return df
            
        except FileNotFoundError:
            print("❌ Файл '../DataBooks.csv' не найден. Используются тестовые данные.")
            return self._create_fallback_data()
        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}. Используются тестовые данные.")
            return self._create_fallback_data()

    def _create_fallback_data(self):
        """Создание тестовых данных при отсутствии CSV файла"""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'title': [
                'Гарри Поттер и философский камень', 
                '1984', 
                'Война и мир', 
                'Мастер и Маргарита', 
                'Анна Каренина',
                'Преступление и наказание',
                'Три товарища',
                'Маленький принц',
                'Алиса в Стране чудес',
                'Улисс'
            ],
            'author': [
                'Джоан Роулинг', 
                'Джордж Оруэлл', 
                'Лев Толстой', 
                'Михаил Булгаков', 
                'Лев Толстой',
                'Федор Достоевский',
                'Эрих Мария Ремарк',
                'Антуан де Сент-Экзюпери',
                'Льюис Кэрролл',
                'Джеймс Джойс'
            ],
            'publisher': ['Росмэн', 'АСТ', 'Эксмо', 'Азбука', 'Эксмо', 'АСТ', 'Эксмо', 'Махаон', 'Росмэн', 'Азбука'],
            'year': [2001, 1949, 1869, 1967, 1877, 1866, 1936, 1943, 1865, 1922],
            'language': ['русский', 'русский', 'русский', 'русский', 'русский', 'русский', 'русский', 'русский', 'русский', 'английский'],
            'age_restriction': ['12+', '16+', '12+', '16+', '12+', '16+', '16+', '6+', '6+', '18+'],
            'genre': ['фэнтези', 'антиутопия', 'роман', 'роман', 'роман', 'роман', 'роман', 'притча', 'фэнтези', 'модернизм'],
            'pages': [400, 328, 1225, 480, 864, 672, 384, 96, 200, 736],
            'has_illustrations': [True, False, False, False, False, False, False, True, True, False]
        })
    
    def process_query(self, query_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка JSON запроса и возврат результатов с текстовым выводом
        """
        try:
            question_type = query_json.get("question_type", "")
            
            # ОБНОВЛЕНИЕ: Логируем входящий запрос для отладки
            print(f"🔍 Получен запрос: question_type='{question_type}', query={json.dumps(query_json, ensure_ascii=False)}")
            
            timestamp = datetime.now().isoformat()
            
            # Обработка в зависимости от типа вопроса
            if question_type == "step_back":
                result = self._handle_step_back(query_json)
            elif question_type == "other":
                result = self._handle_other(query_json)
            elif question_type == "recommendation":
                result = self._handle_recommendation(query_json)
            elif question_type == "search":
                result = self._handle_search(query_json)
            elif question_type == "comparison":
                result = self._handle_comparison(query_json)
            elif question_type == "general":
                result = self._handle_general(query_json)
            else:
                # ОБНОВЛЕНИЕ: Пытаемся определить тип по содержимому
                determined_type = self._determine_question_type(query_json)
                print(f"🔍 Автоопределение типа: '{determined_type}'")
                if determined_type:
                    query_json["question_type"] = determined_type
                    return self.process_query(query_json)  # Рекурсивно обрабатываем с правильным типом
                result = self._create_response([], "Неизвестный тип запроса")
            
            # Сохраняем в историю
            history_entry = {
                "timestamp": timestamp,
                "query": query_json,
                "result": result
            }
            self.query_history.append(history_entry)
            
            # Сохраняем последние результаты
            if "books" in result and result["books"]:
                self.last_results = result["books"]
            
            return result
                
        except Exception as e:
            error_result = self._create_response([], f"Ошибка обработки запроса: {str(e)}")
            
            # Сохраняем ошибку в историю
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query_json,
                "result": error_result
            }
            self.query_history.append(history_entry)
            
            return error_result

    def _determine_question_type(self, query_json: Dict[str, Any]) -> str:
        """
        Определение типа вопроса по содержимому
        """
        # Если есть likes/dislikes - это рекомендации
        feedback = query_json.get("feedback", {})
        if feedback.get("likes") or feedback.get("dislikes"):
            return "recommendation"
        
        # Если есть сравнение книг
        compare = query_json.get("compare", {})
        if compare.get("count_books") and int(compare.get("count_books", 0)) >= 2:
            return "comparison"
        
        # Если есть фильтры - это поиск
        filters = query_json.get("filter", {})
        if any(filters.values()):
            return "search"
        
        # Специальные команды
        if query_json.get("step_back"):
            return "step_back"
        
        if query_json.get("num_question"):
            return "other"
        
        return ""
    
    def _handle_step_back(self, query_json: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка шага назад по истории"""
        step_back = query_json.get("step_back", "")
        
        if step_back == "1":  # Начать сначала
            self.current_filters = {}
            return self._create_response([], "История очищена. Начните новый поиск.", clear_filters=True)
        elif step_back == "-1":  # Шаг назад
            if len(self.query_history) > 1:
                # Удаляем текущий запрос из истории
                current = self.query_history.pop()
                prev_query = self.query_history[-1]["query"]
                
                # Восстанавливаем фильтры из предыдущего запроса
                if "filter" in prev_query:
                    self.current_filters = prev_query["filter"].copy()
                
                return self.process_query(prev_query)
            else:
                return self._create_response([], "Нет предыдущих запросов")
        else:
            return self._create_response([], "Неверное значение step_back")
    
    def _handle_other(self, query_json: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка общих вопросов"""
        num_question = query_json.get("num_question", "")
        
        responses = {
            "вопрос1": "Привет! Я ваш книжный ассистент. Задайте мне вопрос о книгах, и я постараюсь помочь.",
            "вопрос2": "Я могу:\n- Искать книги по названию, автору, жанру\n- Рекомендовать книги на основе ваших предпочтений\n- Сравнивать книги по различным параметрам\n- Отвечать на вопросы о конкретных книгах",
            "вопрос3": "Всегда пожалуйста! Рад, что смог помочь. Если понадобятся ещё рекомендации — просто спросите.",
            "вопрос4": "Я помогу вам:\n- Найти книгу по названию или автору\n- Подобрать книгу по жанру или году\n- Сравнить книги\n- Узнать о книге подробнее\n\nПросто задайте вопрос!",
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
            message = self._format_recommendation_message(similar_books, likes)
            return self._create_response(similar_books, message)
        
        # Если есть дизлайки, исключаем похожие
        dislikes = feedback.get("dislikes", [])
        if dislikes:
            filtered_books = self._exclude_similar_books(dislikes, filtered_books)
        
        message = self._format_search_message(filtered_books)
        return self._create_response(filtered_books.to_dict('records'), message)
    
    def _handle_search(self, query_json: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка поисковых запросов"""
        filters = query_json.get("filter", {})
        
        # Обновляем текущие фильтры
        self.current_filters.update(filters)
        
        # Применяем фильтры
        filtered_books = self._apply_filters(self.df, self.current_filters)
        
        message = self._format_search_message(filtered_books)
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
            message = self._format_comparison_message(comparison_result)
            return self._create_response(comparison_result, message)
        else:
            return self._create_response([], "Не найдено достаточного количества книг для сравнения")
    
    def _handle_general(self, query_json: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка общих вопросов о книгах"""
        filters = query_json.get("filter", {})
        
        # Для общих вопросов обычно ищем одну конкретную книгу
        filtered_books = self._apply_filters(self.df, filters)
        
        if len(filtered_books) == 1:
            message = self._format_book_details(filtered_books.iloc[0])
            return self._create_response([filtered_books.iloc[0].to_dict()], message)
        elif len(filtered_books) > 1:
            message = f"Найдено несколько книг ({len(filtered_books)}). Уточните запрос.\n\n"
            message += self._format_books_list(filtered_books)
            return self._create_response(filtered_books.to_dict('records'), message)
        else:
            return self._create_response([], "Книга не найдена")
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Применение фильтров к DataFrame"""
        filtered_df = df.copy()
        
        # Фильтрация по автору (массив)
        authors = filters.get("author", [])
        if authors and any(authors):
            author_condition = False
            for author in authors:
                if author:
                    author_condition |= filtered_df['author'].str.lower().str.contains(author.lower(), na=False)
            filtered_df = filtered_df[author_condition]
        
        # Фильтрация по издателю (массив)
        publishers = filters.get("publisher", [])
        if publishers and any(publishers):
            publisher_condition = False
            for publisher in publishers:
                if publisher:
                    publisher_condition |= filtered_df['publisher'].str.lower().str.contains(publisher.lower(), na=False)
            filtered_df = filtered_df[publisher_condition]
        
        # Фильтрация по году
        year_from = filters.get("year_from")
        if year_from:
            try:
                filtered_df = filtered_df[filtered_df['year'] >= int(year_from)]
            except:
                pass
        
        year_to = filters.get("year_to")
        if year_to:
            try:
                filtered_df = filtered_df[filtered_df['year'] <= int(year_to)]
            except:
                pass
        
        # Фильтрация по языку (массив)
        languages = filters.get("language", [])
        if languages and any(languages):
            language_condition = False
            for language in languages:
                if language:
                    language_condition |= filtered_df['language'].str.lower().str.contains(language.lower(), na=False)
            filtered_df = filtered_df[language_condition]
        
        # Фильтрация по возрастному ограничению (массив)
        age_restrictions = filters.get("age_restriction", [])
        if age_restrictions and any(age_restrictions):
            age_condition = False
            for age in age_restrictions:
                if age:
                    age_condition |= filtered_df['age_restriction'].str.contains(age, na=False)
            filtered_df = filtered_df[age_condition]
        
        # Фильтрация по жанру (массив)
        genres = filters.get("genre", [])
        if genres and any(genres):
            genre_condition = False
            for genre in genres:
                if genre:
                    genre_condition |= filtered_df['genre'].str.lower().str.contains(genre.lower(), na=False)
            filtered_df = filtered_df[genre_condition]
        
        # Фильтрация по количеству страниц
        pages_from = filters.get("pages_from")
        if pages_from:
            try:
                filtered_df = filtered_df[filtered_df['pages'] >= int(pages_from)]
            except:
                pass
        
        pages_to = filters.get("pages_to")
        if pages_to:
            try:
                filtered_df = filtered_df[filtered_df['pages'] <= int(pages_to)]
            except:
                pass
        
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
                    existing = self.current_filters.get(key, [])
                    if isinstance(existing, list):
                        self.current_filters[key] = list(set(existing + [v for v in value if v]))
                    else:
                        self.current_filters[key] = value
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
                    # Добавляем автора в приоритеты
                    if 'author' not in self.current_filters:
                        self.current_filters['author'] = []
                    author_list = self.current_filters['author']
                    if not isinstance(author_list, list):
                        author_list = [author_list]
                    if book['author'] not in author_list:
                        author_list.append(book['author'])
                        self.current_filters['author'] = author_list
    
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
    
    def _create_response(self, books: List[Dict[str, Any]], message: str, clear_filters: bool = False) -> Dict[str, Any]:
        """Создание стандартного ответа"""
        if clear_filters:
            self.current_filters = {}
            
        return {
            "success": True,
            "message": message,
            "books": books,
            "total_count": len(books),
            "current_filters": self.current_filters,
            "history_count": len(self.query_history)
        }
    
    def _format_search_message(self, filtered_books: pd.DataFrame) -> str:
        """Форматирование сообщения для поисковых запросов"""
        count = len(filtered_books)
        
        if count == 0:
            return "По вашему запросу ничего не найдено. Попробуйте изменить критерии поиска."
        
        message = f"📚 Найдено книг: {count}\n\n"
        
        # Показываем активные фильтры
        active_filters = []
        for key, value in self.current_filters.items():
            if value:
                if isinstance(value, list) and any(value):
                    active_filters.append(f"{key}: {', '.join([str(v) for v in value if v])}")
                elif value:
                    active_filters.append(f"{key}: {value}")
        
        if active_filters:
            message += f"🔍 Активные фильтры: {', '.join(active_filters)}\n\n"
        
        # Добавляем список книг
        message += self._format_books_list(filtered_books)
        
        return message
    
    def _format_recommendation_message(self, books: List[Dict[str, Any]], likes: List[str]) -> str:
        """Форматирование сообщения для рекомендаций"""
        count = len(books)
        
        if count == 0:
            return "К сожалению, не удалось найти подходящих рекомендаций. Попробуйте изменить критерии."
        
        message = f"📖 Рекомендации на основе ваших предпочтений\n\n"
        
        if likes:
            message += f"👍 Вам нравятся: {', '.join(likes)}\n\n"
        
        message += f"📚 Найдено рекомендаций: {count}\n\n"
        message += self._format_books_list(books, is_recommendation=True)
        
        return message
    
    def _format_comparison_message(self, books: List[Dict[str, Any]]) -> str:
        """Форматирование сообщения для сравнения книг"""
        if len(books) < 2:
            return "Недостаточно книг для сравнения."
        
        message = "📊 Сравнение книг:\n\n"
        
        for i, book in enumerate(books, 1):
            message += f"📖 Книга {i}:\n"
            message += self._format_book_details(book, short=True) + "\n\n"
        
        # Сравниваем основные параметры
        message += "⚖️ Сравнение по параметрам:\n"
        
        # Годы издания
        years = [book.get('year', '') for book in books]
        message += f"• Год издания: {', '.join(map(str, years))}\n"
        
        # Количество страниц
        pages = [book.get('pages', '') for book in books]
        message += f"• Страниц: {', '.join(map(str, pages))}\n"
        
        # Возрастные ограничения
        ages = [book.get('age_restriction', '') for book in books]
        message += f"• Возраст: {', '.join(ages)}\n"
        
        # Наличие иллюстраций
        illustrations = ["Да" if book.get('has_illustrations') else "Нет" for book in books]
        message += f"• Иллюстрации: {', '.join(illustrations)}\n"
        
        return message
    
    def _format_book_details(self, book: Dict[str, Any], short: bool = False) -> str:
        """Форматирование деталей книги"""
        if short:
            return (f"«{book.get('title', '')}» - {book.get('author', '')} "
                   f"({book.get('year', '')}), {book.get('pages', '')} стр.")
        
        details = f"📘 «{book.get('title', '')}»\n"
        details += f"✍️ Автор: {book.get('author', '')}\n"
        details += f"🏢 Издатель: {book.get('publisher', '')}\n"
        details += f"📅 Год: {book.get('year', '')}\n"
        details += f"🌐 Язык: {book.get('language', '')}\n"
        details += f"🎭 Жанр: {book.get('genre', '')}\n"
        details += f"👶 Возраст: {book.get('age_restriction', '')}\n"
        details += f"📄 Страниц: {book.get('pages', '')}\n"
        details += f"🖼️ Иллюстрации: {'Да' if book.get('has_illustrations') else 'Нет'}\n"
        
        return details
    
    def _format_books_list(self, books, is_recommendation: bool = False) -> str:
        """Форматирование списка книг"""
        if isinstance(books, pd.DataFrame):
            books_list = books.to_dict('records')
        else:
            books_list = books
            
        if not books_list:
            return "Книги не найдены."
        
        message = ""
        for i, book in enumerate(books_list[:10], 1):  # Ограничиваем 10 книгами
            emoji = "⭐" if is_recommendation else f"{i}."
            message += f"{emoji} «{book.get('title', '')}» - {book.get('author', '')} "
            message += f"({book.get('year', '')}), {book.get('pages', '')} стр.\n"
            
            # Для рекомендаций добавляем причину рекомендации
            if is_recommendation and i <= 3:
                message += f"   💡 Похоже на ваши предпочтения\n"
            
            message += "\n"
        
        if len(books_list) > 10:
            message += f"... и еще {len(books_list) - 10} книг\n"
        
        return message
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Получение истории запросов"""
        return list(self.query_history)
    
    def get_current_filters(self) -> Dict[str, Any]:
        """Получение текущих фильтров"""
        return self.current_filters
    
    def get_last_results(self) -> List[Dict[str, Any]]:
        """Получение последних результатов"""
        return self.last_results


# Пример использования с красивым выводом
def main():
    # Инициализируем систему
    recommender = BookRecommendationSystem()
    
    print("=" * 60)
    print("📚 КНИЖНАЯ РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА")
    print("=" * 60)
    
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
            "compare": {"count_books": "", "title1": "", "author1": "", "title2": "", "author2": ""},
            "feedback": {"likes": [], "dislikes": []},
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
            "compare": {"count_books": "", "title1": "", "author1": "", "title2": "", "author2": ""},
            "feedback": {
                "likes": ["Гарри Поттер и философский камень"],
                "dislikes": []
            },
            "num_question": "",
            "step_back": ""
        },
        {
            "question_type": "comparison",
            "filter": {},
            "compare": {
                "count_books": "2",
                "title1": "Война и мир",
                "author1": "Лев Толстой",
                "title2": "Анна Каренина", 
                "author2": "Лев Толстой"
            },
            "feedback": {"likes": [], "dislikes": []},
            "num_question": "",
            "step_back": ""
        }
    ]
    
    # Обрабатываем запросы
    for i, query in enumerate(test_queries, 1):
        print(f"\n🎯 ЗАПРОС {i}:")
        print("-" * 40)
        
        result = recommender.process_query(query)
        
        # Выводим текстовый результат
        print(result["message"])
        print("-" * 40)
        
        # Показываем текущие фильтры
        current_filters = recommender.get_current_filters()
        if any(current_filters.values()):
            print("🔍 Текущие фильтры:", {k: v for k, v in current_filters.items() if v})
        
        # Показываем историю
        history = recommender.get_history()
        print(f"📖 История запросов: {len(history)}")
    
    # Демонстрация истории
    print("\n" + "=" * 60)
    print("🕐 ПОСЛЕДНИЕ ЗАПРОСЫ:")
    print("=" * 60)
    
    history = recommender.get_history()
    for i, entry in enumerate(history, 1):
        print(f"{i}. [{entry['timestamp'][11:19]}] {entry['query']['question_type']} - {entry['result']['total_count']} книг")


if __name__ == "__main__":
    main()