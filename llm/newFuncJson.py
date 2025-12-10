import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class BookAssistant:
    def __init__(self, data_path='../DataBooks.csv'):
        # Загружаем датасет из файла
        self.df = pd.read_csv(data_path)
        self.metrics = BookDistanceMetrics(self.df)
        self.history = []  # история запросов для шага назад
        
    def process_request(self, request_json):
        """Основной метод обработки входящего JSON запроса"""
        
        # Сохраняем запрос в историю
        self.history.append(request_json.copy())
        
        # Определяем тип запроса
        question_type = self._determine_question_type(request_json)
        
        # Обрабатываем в зависимости от типа
        if question_type == "recommendation":
            return self._handle_recommendation(request_json)
        elif question_type == "search":
            return self._handle_search(request_json)
        elif question_type == "compare":
            return self._handle_compare(request_json)
        elif question_type == "step_back":
            return self._handle_step_back(request_json)
        else:
            return self._handle_other(request_json)
    
    def _determine_question_type(self, request):
        """Определяет тип вопроса на основе заполненных полей"""
        
        # Шаг назад имеет высший приоритет
        if request.get("step_back"):
            return "step_back"
            
        # Проверяем рекомендации по фидбеку
        feedback = request.get("feedback", {})
        if feedback.get("likes") or feedback.get("dislikes"):
            return "recommendation"
            
        # Проверяем сравнение
        compare = request.get("compare", {})
        if (compare.get("title1") or compare.get("title2") or 
            compare.get("count_books")):
            return "compare"
            
        # Проверяем поиск по фильтрам
        filter_data = request.get("filter", {})
        if any(self._is_filter_active(filter_data, key) for key in filter_data):
            return "search"
            
        # Иное по умолчанию
        return "other"
    
    def _is_filter_active(self, filter_data, key):
        """Проверяет, активен ли фильтр"""
        value = filter_data.get(key)
        if value is None:
            return False
        if isinstance(value, list):
            return len(value) > 0
        if isinstance(value, str):
            return value.strip() != ""
        return bool(value)
    
    def _handle_recommendation(self, request):
        """Обработка рекомендательных запросов"""
        feedback = request.get("feedback", {})
        
        # Обеспечиваем, что likes и dislikes являются списками
        likes = feedback.get("likes", [])
        if isinstance(likes, str):
            likes = [likes] if likes.strip() else []
            
        dislikes = feedback.get("dislikes", [])
        if isinstance(dislikes, str):
            dislikes = [dislikes] if dislikes.strip() else []
        
        # Находим похожие книги на основе лайков
        similar_books = []
        for liked_book in likes:
            book_idx = self._find_book_index(liked_book)
            if book_idx is not None:
                similar = self.metrics.get_similar_books(book_idx, n=10)
                similar_books.extend(similar)
        
        # Исключаем книги из дизлайков
        dislike_indices = [self._find_book_index(book) for book in dislikes]
        dislike_indices = [idx for idx in dislike_indices if idx is not None]
        
        # Фильтруем и ранжируем рекомендации
        filtered_recommendations = []
        for book_idx, score in similar_books:
            if book_idx not in dislike_indices:
                book_data = self.df.iloc[book_idx]
                filtered_recommendations.append({
                    'title': book_data['title'],
                    'author': book_data['author'],
                    'genre': book_data['genre'],
                    'year': book_data['year'],
                    'similarity_score': round(score, 3)
                })
        
        # Убираем дубликаты и сортируем по релевантности
        unique_recommendations = []
        seen_titles = set()
        for rec in filtered_recommendations:
            if rec['title'] not in seen_titles:
                unique_recommendations.append(rec)
                seen_titles.add(rec['title'])
        
        unique_recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return {
            "type": "recommendation",
            "message": f"На основе ваших предпочтений рекомендую следующие книги:",
            "books": unique_recommendations[:5]  # топ-5 рекомендаций
        }
    
    def _handle_search(self, request):
        """Обработка поисковых запросов с фильтрами"""
        filter_data = request.get("filter", {})
        filtered_books = self.df.copy()
        
        # Применяем фильтры с проверкой типов данных
        if filter_data.get("author"):
            authors = filter_data["author"]
            # Преобразуем строку в список, если необходимо
            if isinstance(authors, str):
                authors = [authors]
            filtered_books = filtered_books[filtered_books['author'].isin(authors)]
            
        if filter_data.get("genre"):
            genres = filter_data["genre"]
            if isinstance(genres, str):
                genres = [genres]
            filtered_books = filtered_books[filtered_books['genre'].isin(genres)]
            
        if filter_data.get("language"):
            languages = filter_data["language"]
            if isinstance(languages, str):
                languages = [languages]
            filtered_books = filtered_books[filtered_books['language'].isin(languages)]
            
        if filter_data.get("year_from"):
            try:
                year_from = int(filter_data["year_from"])
                filtered_books = filtered_books[filtered_books['year'] >= year_from]
            except (ValueError, TypeError):
                pass
            
        if filter_data.get("year_to"):
            try:
                year_to = int(filter_data["year_to"])
                filtered_books = filtered_books[filtered_books['year'] <= year_to]
            except (ValueError, TypeError):
                pass
            
        if filter_data.get("pages_from"):
            try:
                pages_from = int(filter_data["pages_from"])
                filtered_books = filtered_books[filtered_books['pages'] >= pages_from]
            except (ValueError, TypeError):
                pass
            
        if filter_data.get("pages_to"):
            try:
                pages_to = int(filter_data["pages_to"])
                filtered_books = filtered_books[filtered_books['pages'] <= pages_to]
            except (ValueError, TypeError):
                pass
            
        if filter_data.get("age_restriction"):
            age_restrictions = filter_data["age_restriction"]
            if isinstance(age_restrictions, str):
                age_restrictions = [age_restrictions]
            
            # Пытаемся преобразовать в числа, если возможно
            numeric_ages = []
            for age in age_restrictions:
                try:
                    numeric_ages.append(int(age))
                except (ValueError, TypeError):
                    pass
            
            if numeric_ages:
                filtered_books = filtered_books[filtered_books['age_restriction'].isin(numeric_ages)]
            
        if filter_data.get("has_illustrations") != "":
            has_illustrations_str = filter_data["has_illustrations"]
            if isinstance(has_illustrations_str, str):
                has_illustrations = has_illustrations_str.lower() == "true"
                filtered_books = filtered_books[filtered_books['has_illustrations'] == has_illustrations]
        
        results = filtered_books.to_dict('records')
        
        return {
            "type": "search",
            "message": f"Найдено книг: {len(results)}",
            "books": results[:20]  # ограничиваем вывод
        }
    
    def _handle_compare(self, request):
        """Обработка запросов на сравнение книг"""
        compare_data = request.get("compare", {})
        
        books_to_compare = []
        
        # Собираем книги для сравнения
        for i in range(1, 3):  # поддерживаем сравнение 2 книг
            title_key = f"title{i}"
            author_key = f"author{i}"
            
            title = compare_data.get(title_key)
            author = compare_data.get(author_key)
            
            if title:
                book_idx = self._find_book_index(title, author)
                if book_idx is not None:
                    book_data = self.df.iloc[book_idx]
                    books_to_compare.append(book_data)
        
        if len(books_to_compare) < 2:
            return {
                "type": "compare",
                "message": "Не найдено enough книг для сравнения",
                "comparison": None
            }
        
        # Сравниваем характеристики
        comparison = []
        for book in books_to_compare:
            comparison.append({
                'title': book['title'],
                'author': book['author'],
                'year': book['year'],
                'pages': book['pages'],
                'genre': book['genre'],
                'language': book['language'],
                'age_restriction': book['age_restriction'],
                'has_illustrations': book['has_illustrations']
            })
        
        # Вычисляем схожесть
        idx1 = self._find_book_index(books_to_compare[0]['title'])
        idx2 = self._find_book_index(books_to_compare[1]['title'])
        similarity = self.metrics.similarity_score(idx1, idx2) if idx1 is not None and idx2 is not None else 0
        
        return {
            "type": "compare",
            "message": f"Схожесть книг: {similarity:.1%}",
            "comparison": comparison,
            "similarity": round(similarity, 3)
        }
    
    def _handle_step_back(self, request):
        """Обработка шага назад по истории"""
        if len(self.history) > 1:
            # Возвращаем предыдущий запрос
            previous_request = self.history[-2]
            return self.process_request(previous_request)
        else:
            return {
                "type": "step_back",
                "message": "Нет предыдущих запросов в истории"
            }
    
    def _handle_other(self, request):
        """Обработка прочих запросов (приветствие, помощь и т.д.)"""
        num_question = request.get("num_question", "")
        
        responses = {
            "1": "Привет! Я ваш книжный ассистент. Задайте мне вопрос о книгах, и я постараюсь помочь: найти книгу, порекомендовать что-то новое или сравнить произведения.",
            "2": "Я могу искать книги по названию, автору, жанру и другим параметрам, рекомендовать книги на основе ваших предпочтений, сравнивать книги и отвечать на вопросы о них. Просто расскажите, что вы ищете!",
            "3": "Всегда пожалуйста! Рад, что смог помочь. Если понадобятся ещё рекомендации — просто спросите.",
            "4": "Я помогу вам: найти книгу по названию или автору, подобрать книгу по жанру или году, сравнить книги, узнать о книге подробнее. Просто задайте вопрос!",
            "5": "До свидания! Возвращайтесь, когда понадобятся новые книжные рекомендации."
        }
        
        message = responses.get(num_question, "Извините, я не могу ответить на ваш вопрос")
        
        return {
            "type": "other",
            "message": message
        }
    
    def _find_book_index(self, title, author=None):
        """Находит индекс книги по названию и автору"""
        # Приводим к строке на случай, если пришло число или другой тип
        title = str(title) if title is not None else ""
        
        mask = self.df['title'].str.contains(title, case=False, na=False)
        if author:
            author = str(author) if author is not None else ""
            mask = mask & self.df['author'].str.contains(author, case=False, na=False)
        
        matches = self.df[mask]
        if len(matches) > 0:
            return matches.index[0]
        return None


class BookDistanceMetrics:
    def __init__(self, df):
        self.df = df
        self._setup_taxonomy_tree()
        self._setup_scalers()
        self.numerical_features = ['year', 'pages']
    
    def _setup_taxonomy_tree(self):
        """Иерархия жанров в виде дерева"""
        self.genre_hierarchy = {
            'художественная': {
                'проза': {
                    'классика': {},
                    'роман': {
                        'любовный_роман': {},
                        'исторический_роман': {}
                    },
                    'фантастика': {
                        'научная_фантастика': {},
                        'антиутопия': {},
                        'фэнтези': {}
                    },
                    'приключения': {},
                    'детектив': {},
                    'сатира': {}
                },
                'поэзия': {},
                'драма': {}
            },
            'нехудожественная': {
                'биография': {},
                'научная': {},
                'популярная_наука': {},
                'история': {},
                'психология': {}
            },
            'детская': {
                'сказка': {},
                'развивающая': {},
                'приключения_детские': {}
            }
        }
        
        # Сопоставление жанров из датасета с узлами дерева
        self.genre_mapping = {
            'классика': 'художественная.проза.классика',
            'антиутопия': 'художественная.проза.фантастика.антиутопия',
            'фэнтези': 'художественная.проза.фантастика.фэнтези',
            'сказка': 'детская.сказка',
            'роман': 'художественная.проза.роман',
            'детектив': 'художественная.проза.детектив',
            'фантастика': 'художественная.проза.фантастика',
            'приключения': 'художественная.проза.приключения',
            'сатира': 'художественная.проза.сатира'
        }
    
    def _setup_scalers(self):
        """Инициализация нормализаторов для числовых признаков"""
        self.scalers = {}
        
        # Год публикации (чем новее, тем лучше)
        years = self.df['year'].values.reshape(-1, 1)
        self.scalers['year'] = MinMaxScaler().fit(years)
        
        # Возрастное ограничение
        age = self.df['age_restriction'].values.reshape(-1, 1)
        self.scalers['age_restriction'] = MinMaxScaler().fit(age)
        
        # Количество страниц
        pages = self.df['pages'].values.reshape(-1, 1)
        self.scalers['pages'] = MinMaxScaler().fit(pages)
    
    def _get_genre_path(self, genre):
        """Получить путь жанра в дереве"""
        return self.genre_mapping.get(genre, '')
    
    def _find_common_ancestor_level(self, path1, path2):
        """Найти уровень общего предка в дереве"""
        if not path1 or not path2:
            return 0
            
        parts1 = path1.split('.')
        parts2 = path2.split('.')
        
        common_level = 0
        for i in range(min(len(parts1), len(parts2))):
            if parts1[i] == parts2[i]:
                common_level += 1
            else:
                break
        return common_level
    
    def taxonomic_distance(self, genre1, genre2, max_depth=4):
        """
        Таксономическое расстояние между жанрами на основе дерева
        Возвращает значение от 0 (одинаковые жанры) до 1 (совсем разные)
        """
        path1 = self._get_genre_path(genre1)
        path2 = self._get_genre_path(genre2)
        
        if not path1 or not path2:
            return 1.0
        
        if path1 == path2:
            return 0.0
        
        common_level = self._find_common_ancestor_level(path1, path2)
        depth1 = len(path1.split('.'))
        depth2 = len(path2.split('.'))
        
        # Формула таксономического расстояния
        distance = (depth1 + depth2 - 2 * common_level) / (2 * max_depth)
        return min(distance, 1.0)
    
    def get_numerical_vector(self, book_idx):
        """Получает вектор числовых признаков для книги"""
        book = self.df.iloc[book_idx]
        vector = []

        for feature in self.numerical_features:
            value = self.scalers[feature].transform([[book[feature]]])[0][0]
            vector.append(value)

        return np.array(vector)
    
    def manhattan_distance(self, i, j):
        """Манхэттенское расстояние"""
        vec1 = self.get_numerical_vector(i)
        vec2 = self.get_numerical_vector(j)
        return np.sum(np.abs(vec1 - vec2))
    
    def categorical_distance(self, val1, val2):
        """
        Расстояние для категориальных признаков
        0 - одинаковые значения, 1 - разные
        """
        return 0.0 if val1 == val2 else 1.0

    def bin_distance(self, val1, val2):
        """
        Расстояние для бинарных признаков
        0 - одинаковые значения, 1 - разные
        """
        return 0.0 if val1 == val2 else 1.0
    
    def composite_distance(self, book1_idx, book2_idx, weights=None):
        """
        Композитное расстояние между двумя книгами
        """
        if weights is None:
            weights = {
                'genre': 0.35,
                'has_illustrations': 0.15,
                'author': 0.2,
                'publisher': 0.05,
                'language': 0.05,
                'numerical': 0.2
            }

        book1 = self.df.iloc[book1_idx]
        book2 = self.df.iloc[book2_idx]
        
        total_distance = 0.0
        
        # Таксономическое расстояние по жанру
        genre_dist = self.taxonomic_distance(book1['genre'], book2['genre'])
        total_distance += weights['genre'] * genre_dist
        
        # Числовые признаки
        numerical_dist = self.manhattan_distance(book1_idx, book2_idx)
        total_distance += weights['numerical'] * numerical_dist
        
        # Категориальные признаки
        author_dist = self.categorical_distance(book1['author'], book2['author'])
        total_distance += weights['author'] * author_dist
        
        publisher_dist = self.categorical_distance(book1['publisher'], book2['publisher'])
        total_distance += weights['publisher'] * publisher_dist
        
        language_dist = self.categorical_distance(book1['language'], book2['language'])
        total_distance += weights['language'] * language_dist

        # Бинарные признаки
        ill_dist = self.bin_distance(book1['has_illustrations'], book2['has_illustrations'])
        total_distance += weights['has_illustrations'] * ill_dist

        return total_distance
    
    def similarity_score(self, book1_idx, book2_idx, weights=None):
        """
        Оценка схожести между книгами (1 - distance)
        """
        distance = self.composite_distance(book1_idx, book2_idx, weights)
        return 1.0 - distance
    
    def get_similar_books(self, book_idx, n=5, weights=None):
        """
        Найти n наиболее похожих книг
        """
        similarities = []
        
        for i in range(len(self.df)):
            if i != book_idx:
                sim = self.similarity_score(book_idx, i, weights)
                similarities.append((i, sim))
        
        # Сортировка по убыванию схожести
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]


# Пример использования
if __name__ == "__main__":
    # Создаем ассистента с загрузкой из DataBooks.csv
    assistant = BookAssistant('../DataBooks.csv')
    
    # Пример запроса на рекомендацию
    recommendation_request = {
        "question_type": "",
        "filter": {"author": [], "publisher": [], "year_from": "", "year_to": "", "language": [], "age_restriction": [], "genre": [], "pages_from": "", "pages_to": "", "has_illustrations": ""},
        "compare": {"count_books": "", "title1": "", "author1": "", "title2": "", "author2": ""},
        "feedback": {"likes": ["Гарри Поттер и философский камень"], "dislikes": []},
        "num_question": "",
        "step_back": ""
    }
    
    result = assistant.process_request(recommendation_request)
    print("Рекомендации:", result)