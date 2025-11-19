import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import requests
import json
import re
import copy
from typing import Dict, List, Any
from BookDistance import BookDistanceMetrics

# ==================== КЛАССЫ ДЛЯ РАБОТЫ С LLM ====================

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "llama3.1:8b-instruct-q4_0"
    
    def check_connection(self) -> bool:
        """Проверка подключения к Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        """Генерация ответа через Ollama API"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            print(f"Ошибка при запросе к Ollama: {e}")
            return None

class QueryClassifier:
    def __init__(self, ollama_client):
        self.ollama_client = ollama_client
    
    def classify_query(self, user_input: str) -> Dict[str, Any]:
        """Классифицирует пользовательский запрос"""
        
        classification_prompt = """Ты - классификатор запросов для книжной рекомендательной системы. 

КАТЕГОРИИ: РЕКОМЕНДАЦИЯ, ПОИСК, СРАВНЕНИЕ, ИНФОРМАЦИЯ, ИСТОРИЯ, ДРУГОЕ

Верни JSON в формате:
"category": "категория", "confidence": 0.95, "reasoning": "объяснение", "requires_follow_up": false

Запрос: {user_input}"""
        print(6)
        
        prompt = classification_prompt.format(user_input=user_input)
        print(prompt)
        print(5)
        
        response = self.ollama_client.generate_response(
            prompt=prompt,
            system_prompt="Ты - точный классификатор запросов. Отвечай ТОЛЬКО в формате JSON."
        )
        print(response)
        if response:
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    classification = json.loads(json_match.group())
                    return classification
            except:
                pass
        
        # Fallback классификация
        return self._fallback_classification(user_input)
    
    def _fallback_classification(self, user_input: str) -> Dict[str, Any]:
        print("fall")
        """Резервная классификация"""
        user_input_lower = user_input.lower()
        
        recommendation_keywords = ['посоветуй', 'рекомендуй', 'подбери', 'что почитать', 'похож', 'нравится']
        search_keywords = ['найди', 'ищи', 'покажи', 'поиск', 'где книга']
        comparison_keywords = ['сравни', 'что лучше', 'чем отличается', 'разница между']
        history_keywords = ['назад', 'отмени', 'верни', 'шаг назад', 'предыдущ']
        
        if any(keyword in user_input_lower for keyword in recommendation_keywords):
            category = "РЕКОМЕНДАЦИЯ"
        elif any(keyword in user_input_lower for keyword in search_keywords):
            category = "ПОИСК"
        elif any(keyword in user_input_lower for keyword in comparison_keywords):
            category = "СРАВНЕНИЕ"
        elif any(keyword in user_input_lower for keyword in history_keywords):
            category = "ИСТОРИЯ"
        elif any(word in user_input_lower for word in ['кто', 'что', 'когда', 'сколько', 'какой', 'какие']):
            category = "ИНФОРМАЦИЯ"
        else:
            category = "ДРУГОЕ"
        
        return {
            "category": category,
            "confidence": 0.7,
            "reasoning": "Автоматическая классификация по ключевым словам",
            "requires_follow_up": category in ["РЕКОМЕНДАЦИЯ", "ПОИСК"]
        }

# ==================== СИСТЕМА РЕКОМЕНДАЦИЙ ====================

class RecommendationEngine:
    def __init__(self, df):
        self.df = df
        self.metrics_full = BookDistanceMetrics(self.df)
        self.metrics_filtered = None
        self.current_filters = {}
        self.liked_books = []
        self.disliked_books = []
    
    def apply_filters(self, filters: Dict) -> pd.DataFrame:
        """Применяет фильтры к данным"""
        filtered_df = self.df.copy()
        
        if 'genres' in filters and filters['genres']:
            filtered_df = filtered_df[filtered_df['genre'].isin(filters['genres'])]
        
        if 'authors' in filters and filters['authors']:
            filtered_df = filtered_df[filtered_df['author'].isin(filters['authors'])]
        
        if 'themes' in filters and filters['themes']:
            # Простая фильтрация по темам через ключевые слова в названии
            theme_conditions = []
            for theme in filters['themes']:
                theme_conditions.append(filtered_df['title'].str.contains(theme, case=False, na=False))
            
            if theme_conditions:
                combined_condition = theme_conditions[0]
                for condition in theme_conditions[1:]:
                    combined_condition = combined_condition | condition
                filtered_df = filtered_df[combined_condition]
        
        self.current_filters = filters
        self.metrics_filtered = BookDistanceMetrics(filtered_df)
        
        return filtered_df
    
    def get_recommendations(self, n_recommendations: int = 10) -> List[Dict]:
        """Получает рекомендации на основе текущих фильтров и предпочтений"""
        if self.metrics_filtered is None or len(self.metrics_filtered.df) == 0:
            filtered_df = self.df
            self.metrics_filtered = BookDistanceMetrics(filtered_df)
        
        # Используем ранее реализованные функции рекомендаций
        try:
            # Если есть лайки, используем рекомендации на основе предпочтений
            if self.liked_books:
                # Преобразуем названия книг в индексы
                liked_indices = []
                for book_title in self.liked_books:
                    matches = self.metrics_filtered.df[self.metrics_filtered.df['title'] == book_title]
                    if not matches.empty:
                        liked_indices.append(matches.index[0])
                
                if liked_indices:
                    recommendations = self._recommend_based_on_likes(
                        liked_indices, n_recommendations
                    )
                    return recommendations
            
            # Иначе используем общие рекомендации из отфильтрованного датасета
            recommendations = []
            n_books = min(n_recommendations, len(self.metrics_filtered.df))
            
            for i in range(n_books):
                book = self.metrics_filtered.df.iloc[i]
                recommendations.append({
                    'title': book['title'],
                    'author': book['author'],
                    'genre': book['genre'],
                    'year': book['year'],
                    'pages': book['pages'],
                    'score': 0.9 - (i * 0.05)
                })
            
            return recommendations
            
        except Exception as e:
            print(f"Ошибка при генерации рекомендаций: {e}")
            return self._get_fallback_recommendations(n_recommendations)
    
    def _recommend_based_on_likes(self, liked_indices: List[int], n_recommendations: int) -> List[Dict]:
        """Рекомендации на основе понравившихся книг"""
        recommendations = []
        
        for liked_idx in liked_indices:
            # Получаем похожие книги для каждой понравившейся
            similar_books = self.metrics_filtered.get_similar_books(
                liked_idx, n_recommendations * 2
            )
            
            for book_idx, similarity in similar_books:
                if book_idx not in liked_indices:  # Исключаем уже понравившиеся
                    book = self.metrics_filtered.df.iloc[book_idx]
                    recommendations.append({
                        'title': book['title'],
                        'author': book['author'],
                        'genre': book['genre'],
                        'year': book['year'],
                        'pages': book['pages'],
                        'score': similarity,
                        'based_on': self.metrics_filtered.df.iloc[liked_idx]['title']
                    })
        
        # Сортируем по схожести и убираем дубликаты
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        seen_titles = set()
        unique_recommendations = []
        
        for rec in recommendations:
            if rec['title'] not in seen_titles:
                seen_titles.add(rec['title'])
                unique_recommendations.append(rec)
            if len(unique_recommendations) >= n_recommendations:
                break
        
        return unique_recommendations
    
    def _get_fallback_recommendations(self, n_recommendations: int) -> List[Dict]:
        """Резервные рекомендации"""
        recommendations = []
        n_books = min(n_recommendations, len(self.df))
        
        for i in range(n_books):
            book = self.df.iloc[i]
            recommendations.append({
                'title': book['title'],
                'author': book['author'],
                'genre': book['genre'],
                'year': book['year'],
                'pages': book['pages'],
                'score': 0.8
            })
        
        return recommendations
    
    def add_like(self, book_title: str):
        """Добавляет книгу в понравившиеся"""
        if book_title not in self.liked_books:
            self.liked_books.append(book_title)
    
    def add_dislike(self, book_title: str):
        """Добавляет книгу в непонравившиеся"""
        if book_title not in self.disliked_books:
            self.disliked_books.append(book_title)
    
    def clear_preferences(self):
        """Очищает предпочтения"""
        self.liked_books = []
        self.disliked_books = []

# ==================== ДИАЛОГОВЫЙ ПРОЦЕССОР ====================

class DialogProcessor:
    def __init__(self, df, ollama_client):
        self.df = df
        self.ollama_client = ollama_client
        self.classifier = QueryClassifier(ollama_client)
        self.recommendation_engine = RecommendationEngine(df)
        
        self.conversation_history = []
        self.last_recommendations = []
    
    def process_message(self, user_input: str) -> Dict[str, Any]:
        """Обрабатывает сообщение пользователя"""
        print("2")
        # Классифицируем запрос
        classification = self.classifier.classify_query(user_input)
        print(classification)
        print(3)
        category = classification['category']
        print(4)
        
        # Добавляем в историю
        self.conversation_history.append({
            'role': 'user',
            'content': user_input,
            'category': category
        })
        
        # Обрабатываем в зависимости от категории
        if category == "РЕКОМЕНДАЦИЯ":
            return self._handle_recommendation(user_input)
        elif category == "ПОИСК":
            return self._handle_search(user_input)
        elif category == "СРАВНЕНИЕ":
            return self._handle_comparison(user_input)
        elif category == "ИНФОРМАЦИЯ":
            return self._handle_information(user_input)
        elif category == "ИСТОРИЯ":
            return self._handle_history(user_input)
        else:
            return self._handle_other(user_input)
    
    def _handle_recommendation(self, user_input: str) -> Dict[str, Any]:
        """Обработка рекомендательных запросов"""
        # Извлекаем сущности через LLM
        entities = self._extract_entities(user_input)
        
        # Применяем фильтры
        filtered_df = self.recommendation_engine.apply_filters(entities)
        
        # Получаем рекомендации
        recommendations = self.recommendation_engine.get_recommendations(10)
        self.last_recommendations = recommendations
        
        # Генерируем ответ
        response_text = self._generate_response(
            user_input, 
            f"Нашел {len(recommendations)} рекомендаций по вашему запросу",
            "recommendation"
        )
        
        return {
            'response': response_text,
            'recommendations': recommendations,
            'type': 'recommendation'
        }
    
    def _handle_search(self, user_input: str) -> Dict[str, Any]:
        """Обработка поисковых запросов"""
        entities = self._extract_entities(user_input)
        filtered_df = self.recommendation_engine.apply_filters(entities)
        results = self.recommendation_engine.get_recommendations(15)
        
        response_text = self._generate_response(
            user_input,
            f"Нашел {len(results)} книг по вашему запросу",
            "search"
        )
        
        return {
            'response': response_text,
            'recommendations': results,
            'type': 'search'
        }
    
    def _handle_comparison(self, user_input: str) -> Dict[str, Any]:
        """Обработка запросов на сравнение"""
        response_text = self._generate_response(
            user_input,
            "Могу сравнить книги, авторов или жанры. Уточните, что именно вы хотите сравнить?",
            "comparison"
        )
        
        return {
            'response': response_text,
            'type': 'comparison'
        }
    
    def _handle_information(self, user_input: str) -> Dict[str, Any]:
        """Обработка информационных запросов"""
        response_text = self._generate_response(
            user_input,
            "Ищу информацию по вашему запросу...",
            "information"
        )
        
        return {
            'response': response_text,
            'type': 'information'
        }
    
    def _handle_history(self, user_input: str) -> Dict[str, Any]:
        """Обработка запросов истории"""
        if len(self.conversation_history) > 1:
            # Возвращаем предыдущие рекомендации
            prev_recommendations = self.last_recommendations
            response_text = "Вернулись к предыдущим рекомендациям"
        else:
            prev_recommendations = []
            response_text = "Это первый запрос, нечего возвращать"
        
        return {
            'response': response_text,
            'recommendations': prev_recommendations,
            'type': 'history'
        }
    
    def _handle_other(self, user_input: str) -> Dict[str, Any]:
        """Обработка прочих запросов"""
        response_text = self._generate_response(
            user_input,
            "Я специализируюсь на помощи с книгами. Могу помочь с поиском, рекомендациями или информацией о книгах!",
            "other"
        )
        
        return {
            'response': response_text,
            'type': 'other'
        }
    
    def _extract_entities(self, user_input: str) -> Dict:
        """Извлечение сущностей из запроса"""
        extraction_prompt = f"""
        Извлеки информацию о книжных предпочтениях из запроса: "{user_input}"
        
        Верни JSON с полями:
        - genres: список жанров
        - authors: список авторов  
        - themes: список тем
        - moods: список настроений
        
        Пример: {{"genres": ["фантастика"], "themes": ["космос"]}}
        """
        
        response = self.ollama_client.generate_response(extraction_prompt)
        if response:
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
        
        return {}
    
    def _generate_response(self, user_input: str, context: str, response_type: str) -> str:
        """Генерация естественного ответа"""
        response_prompt = f"""
        Пользователь сказал: "{user_input}"
        Контекст: {context}
        
        Сгенерируй краткий, дружелюбный ответ на русском (1-2 предложения).
        """
        
        response = self.ollama_client.generate_response(response_prompt)
        return response if response else context
    
    def add_feedback(self, book_title: str, like: bool):
        """Добавляет обратную связь по книге"""
        if like:
            self.recommendation_engine.add_like(book_title)
        else:
            self.recommendation_engine.add_dislike(book_title)

# ==================== ГРАФИЧЕСКИЙ ИНТЕРФЕЙС ====================

class BookRecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Умный книжный помощник")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f5f5f5')
        
        # Загрузка данных
        try:
            self.df = pd.read_csv('../DataBooks.csv')
        except:
            self.df = self._create_sample_data()
        
        # Инициализация компонентов
        self.ollama_client = OllamaClient()
        self.dialog_processor = DialogProcessor(self.df, self.ollama_client)
        
        self.setup_gui()
        self.check_connection()
    
    def _create_sample_data(self):
        """Создание демо-данных"""
        data = {
            'title': [
                'Мастер и Маргарита', 'Преступление и наказание', '1984', 
                'Гарри Поттер и философский камень', 'Война и мир',
                'Три товарища', 'Маленький принц', 'Анна Каренина',
                'Сто лет одиночества', 'Убить пересмешника'
            ],
            'author': [
                'Михаил Булгаков', 'Федор Достоевский', 'Джордж Оруэлл',
                'Джоан Роулинг', 'Лев Толстой', 'Эрих Мария Ремарк',
                'Антуан де Сент-Экзюпери', 'Лев Толстой', 'Габриэль Гарсиа Маркес',
                'Харпер Ли'
            ],
            'genre': [
                'роман', 'роман', 'антиутопия', 'фэнтези', 'роман',
                'роман', 'притча', 'роман', 'магический реализм', 'роман'
            ],
            'year': [1967, 1866, 1949, 1997, 1869, 1936, 1943, 1877, 1967, 1960],
            'pages': [384, 672, 328, 320, 1225, 384, 96, 864, 416, 281]
        }
        return pd.DataFrame(data)
    
    def setup_gui(self):
        """Настройка графического интерфейса"""
        # Главный контейнер
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Заголовок и статус
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill='x', pady=(0, 10))
        
        title_label = ttk.Label(
            header_frame, 
            text="📚 Умный книжный помощник", 
            font=('Arial', 16, 'bold')
        )
        title_label.pack(side='left')
        
        self.status_label = ttk.Label(
            header_frame, 
            text="🔴 Проверка подключения...",
            font=('Arial', 10)
        )
        self.status_label.pack(side='right')
        
        # Основная область с диалогом и рекомендациями
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill='both', expand=True)
        
        # Левая панель - диалог
        self.setup_dialog_panel(content_frame)
        
        # Правая панель - рекомендации
        self.setup_recommendations_panel(content_frame)
        
        # Панель ввода
        self.setup_input_panel(main_frame)
    
    def setup_dialog_panel(self, parent):
        """Настройка панели диалога"""
        dialog_frame = ttk.LabelFrame(parent, text="💬 Диалог с помощником", padding=10)
        dialog_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Область диалога
        self.dialog_text = scrolledtext.ScrolledText(
            dialog_frame,
            wrap=tk.WORD,
            font=('Arial', 10),
            bg='#fafafa',
            height=20
        )
        self.dialog_text.pack(fill='both', expand=True)
        self.dialog_text.config(state=tk.DISABLED)
        
        # Быстрые действия
        actions_frame = ttk.Frame(dialog_frame)
        actions_frame.pack(fill='x', pady=(10, 0))
        
        actions = [
            ("↶ Шаг назад", self.undo_action),
            ("🔄 Сбросить", self.reset_dialog),
            ("📊 Статистика", self.show_stats)
        ]
        
        for text, command in actions:
            ttk.Button(actions_frame, text=text, command=command).pack(side='left', padx=2)
        
        # Добавляем приветственное сообщение
        self.add_message("🤖 Помощник", "Привет! Я помогу вам найти интересные книги. Спросите меня о книгах, авторах или жанрах!", "system")
    
    def setup_recommendations_panel(self, parent):
        """Настройка панели рекомендаций"""
        rec_frame = ttk.LabelFrame(parent, text="📚 Рекомендации", padding=10)
        rec_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Таблица рекомендаций
        columns = ('#1', '#2', '#3', '#4', '#5')
        self.rec_tree = ttk.Treeview(
            rec_frame, 
            columns=columns, 
            show='headings',
            height=15
        )
        
        # Заголовки колонок
        self.rec_tree.heading('#1', text='Название')
        self.rec_tree.heading('#2', text='Автор')
        self.rec_tree.heading('#3', text='Жанр')
        self.rec_tree.heading('#4', text='Год')
        self.rec_tree.heading('#5', text='Оценка')
        
        # Настройка колонок
        self.rec_tree.column('#1', width=200)
        self.rec_tree.column('#2', width=150)
        self.rec_tree.column('#3', width=100)
        self.rec_tree.column('#4', width=60)
        self.rec_tree.column('#5', width=80)
        
        self.rec_tree.pack(fill='both', expand=True)
        
        # Кнопки взаимодействия с рекомендациями
        feedback_frame = ttk.Frame(rec_frame)
        feedback_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(feedback_frame, text="👍 Понравилось", 
                  command=lambda: self.add_feedback(True)).pack(side='left', padx=2)
        ttk.Button(feedback_frame, text="👎 Не понравилось", 
                  command=lambda: self.add_feedback(False)).pack(side='left', padx=2)
        ttk.Button(feedback_frame, text="📋 Показать все", 
                  command=self.show_all_books).pack(side='right', padx=2)
    
    def setup_input_panel(self, parent):
        """Настройка панели ввода"""
        input_frame = ttk.Frame(parent)
        input_frame.pack(fill='x', pady=(10, 0))
        
        # Поле ввода
        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(
            input_frame,
            textvariable=self.input_var,
            font=('Arial', 12)
        )
        self.input_entry.pack(fill='x', side='left', expand=True, padx=(0, 5))
        self.input_entry.bind('<Return>', lambda e: self.send_message())
        
        # Кнопка отправки
        self.send_btn = ttk.Button(
            input_frame,
            text="Отправить ✉️",
            command=self.send_message
        )
        self.send_btn.pack(side='right')
        
        # Подсказки
        hints_frame = ttk.Frame(parent)
        hints_frame.pack(fill='x', pady=(5, 0))
        
        ttk.Label(hints_frame, text="💡 Примеры:", font=('Arial', 9)).pack(anchor='w')
        examples = [
            "• 'Посоветуй фантастику про космос'",
            "• 'Найди книги Толстого'", 
            "• 'Что почитать если нравится детектив?'",
            "• 'Шаг назад' для возврата"
        ]
        
        for example in examples:
            example_label = ttk.Label(
                hints_frame, 
                text=example, 
                font=('Arial', 8),
                foreground='gray'
            )
            example_label.pack(anchor='w')
    
    def check_connection(self):
        """Проверка подключения к Ollama"""
        if self.ollama_client.check_connection():
            self.status_label.config(text="🟢 Подключено к AI")
            self.send_btn.config(state='normal')
        else:
            self.status_label.config(text="🔴 Ошибка подключения")
            self.send_btn.config(state='disabled')
            messagebox.showerror(
                "Ошибка подключения", 
                "Не удалось подключиться к Ollama.\nУбедитесь, что сервер запущен на localhost:11434"
            )
    
    def send_message(self):
        """Отправка сообщения пользователя"""
        user_input = self.input_var.get().strip()
        if not user_input:
            return
        
        # Очищаем поле ввода
        self.input_var.set("")
        
        # Добавляем сообщение пользователя в диалог
        self.add_message("👤 Вы", user_input, "user")
        
        # Обрабатываем сообщение
        try:
            print("1")
            response = self.dialog_processor.process_message(user_input)
            
            # Добавляем ответ помощника
            self.add_message("🤖 Помощник", response['response'], "assistant")
            
            # Обновляем рекомендации если есть
            if 'recommendations' in response:
                self.update_recommendations(response['recommendations'])
                
        except Exception as e:
            error_msg = f"Извините, произошла ошибка: {str(e)}"
            print(e)
            self.add_message("🤖 Помощник", error_msg, "error")
    
    def add_message(self, sender: str, message: str, msg_type: str):
        """Добавление сообщения в диалог"""
        self.dialog_text.config(state=tk.NORMAL)
        
        # Цвета для разных типов сообщений
        colors = {
            'user': '#007acc',
            'assistant': '#2e7d32', 
            'system': '#666666',
            'error': '#d32f2f'
        }
        
        color = colors.get(msg_type, '#000000')
        
        # Добавляем отправителя
        self.dialog_text.insert(tk.END, f"{sender}: ", f"bold_{msg_type}")
        self.dialog_text.tag_configure(f"bold_{msg_type}", foreground=color, font=('Arial', 10, 'bold'))
        
        # Добавляем сообщение
        self.dialog_text.insert(tk.END, f"{message}\n\n", msg_type)
        self.dialog_text.tag_configure(msg_type, foreground=color)
        
        self.dialog_text.see(tk.END)
        self.dialog_text.config(state=tk.DISABLED)
    
    def update_recommendations(self, recommendations: List[Dict]):
        """Обновление списка рекомендаций"""
        # Очищаем текущий список
        for item in self.rec_tree.get_children():
            self.rec_tree.delete(item)
        
        # Добавляем новые рекомендации
        for i, rec in enumerate(recommendations):
            score = f"{rec.get('score', 0):.2f}"
            self.rec_tree.insert('', 'end', values=(
                rec['title'],
                rec['author'],
                rec['genre'],
                rec['year'],
                score
            ))
    
    def add_feedback(self, like: bool):
        """Добавление обратной связи по выбранной книге"""
        selection = self.rec_tree.selection()
        if not selection:
            messagebox.showwarning("Внимание", "Выберите книгу из списка рекомендаций")
            return
        
        item = self.rec_tree.item(selection[0])
        book_title = item['values'][0]
        
        self.dialog_processor.add_feedback(book_title, like)
        
        action = "понравилась" if like else "не понравилась"
        self.add_message("🤖 Помощник", f"Запомнил, что книга '{book_title}' {action} вам!", "system")
    
    def undo_action(self):
        """Шаг назад"""
        response = self.dialog_processor.process_message("шаг назад")
        self.add_message("🤖 Помощник", response['response'], "system")
        
        if 'recommendations' in response:
            self.update_recommendations(response['recommendations'])
    
    def reset_dialog(self):
        """Сброс диалога"""
        self.dialog_processor = DialogProcessor(self.df, self.ollama_client)
        self.dialog_text.config(state=tk.NORMAL)
        self.dialog_text.delete(1.0, tk.END)
        self.dialog_text.config(state=tk.DISABLED)
        
        self.rec_tree.delete(*self.rec_tree.get_children())
        self.add_message("🤖 Помощник", "Диалог сброшен. Чем могу помочь?", "system")
    
    def show_stats(self):
        """Показать статистику"""
        total_books = len(self.df)
        genres = self.df['genre'].value_counts()
        
        stats_text = f"📊 Статистика библиотеки:\n"
        stats_text += f"• Всего книг: {total_books}\n"
        stats_text += f"• Популярные жанры:\n"
        
        for genre, count in genres.head(5).items():
            stats_text += f"  - {genre}: {count} книг\n"
        
        self.add_message("🤖 Помощник", stats_text, "system")
    
    def show_all_books(self):
        """Показать все книги"""
        all_books = []
        for i in range(min(20, len(self.df))):
            book = self.df.iloc[i]
            all_books.append({
                'title': book['title'],
                'author': book['author'],
                'genre': book['genre'],
                'year': book['year'],
                'pages': book['pages'],
                'score': 0.8
            })
        
        self.update_recommendations(all_books)
        self.add_message("🤖 Помощник", f"Показал {len(all_books)} книг из библиотеки", "system")

# ==================== ЗАПУСК ПРИЛОЖЕНИЯ ====================

if __name__ == "__main__":
    root = tk.Tk()
    app = BookRecommendationApp(root)
    root.mainloop()

