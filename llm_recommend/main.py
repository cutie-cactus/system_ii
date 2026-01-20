"""
Главный модуль системы рекомендаций книг с поддержкой истории
"""
import sys
from typing import Dict, Any
from config import Config
from data_loader import BookDataLoader
from neural_parser import NeuralBookParser
from query_processor import QueryProcessor
from book_recommender import BookRecommender
from book_metrics import BookDistanceMetrics

class BookRecommendationSystem:
    def __init__(self):
        self.data_loader = None
        self.neural_parser = None
        self.query_processor = None
        self.recommender = None
        self.initialized = False
        
    def initialize(self) -> bool:
        """Инициализация системы"""
        try:
            print("=" * 60)
            print("📚 СИСТЕМА РЕКОМЕНДАЦИЙ КНИГ С ИСТОРИЕЙ")
            print("=" * 60)
            
            # 1. Инициализация нейросети
            print("\n🧠 ИНИЦИАЛИЗАЦИЯ НЕЙРОСЕТИ...")
            self.neural_parser = NeuralBookParser()
            if not self.neural_parser.initialize():
                print("❌ Ошибка инициализации нейросети")
                return False
            
            # 2. Загрузка данных
            print("\n📊 ЗАГРУЗКА ДАННЫХ...")
            self.data_loader = BookDataLoader(Config.DATA_PATH)
            self.data_loader.load_data()
            
            # 3. Инициализация обработчика запросов с историей
            print("\n⚙️ ИНИЦИАЛИЗАЦИЯ ОБРАБОТЧИКА С ИСТОРИЕЙ...")
            self.query_processor = QueryProcessor(self.data_loader)
            self.query_processor.initialize_metrics()
            
            # 4. Инициализация рекомендательной системы
            print("\n🎯 ИНИЦИАЛИЗАЦИЯ РЕКОМЕНДАТЕЛЬНОЙ СИСТЕМЫ...")
            self.recommender = BookRecommender(
                self.query_processor.metrics_full,
                self.query_processor.metrics_filtered
            )
            
            self.initialized = True
            print("\n✅ СИСТЕМА УСПЕШНО ИНИЦИАЛИЗИРОВАНА!")
            print("   Поддерживается история на 5 шагов назад")
            print("   Используйте 'назад' для возврата, 'заново' для сброса")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"❌ Ошибка инициализации системы: {e}")
            return False
    
    def process_user_query(self, query: str) -> Dict[str, Any]:
        """
        Обработка пользовательского запроса с учетом истории
        
        Returns:
            Словарь с результатами обработки
        """
        if not self.initialized:
            return {"error": "Система не инициализирована"}
        
        result = {
            "success": False,
            "query": query,
            "query_type": "",
            "message": "",
            "state_info": None,
            "data": None
        }
        
        try:
            # 1. Парсинг запроса нейросетью
            parsed_query = self.neural_parser.parse_query(query)
            
            if not parsed_query or not parsed_query.get('question_type'):
                result["message"] = "Извините, я не могу ответить на ваш вопрос"
                return result
            
            result["query_type"] = parsed_query.get('question_type', '')
            
            # 2. Обработка распарсенного запроса с учетом истории
            processed = self.query_processor.process_query(parsed_query)
            
            # 3. Получаем информацию о текущем состоянии
            result["state_info"] = self.query_processor.get_current_state_info()
            result["history_info"] = processed.get("history_info", {})
            
            # 4. Проверяем на нераспознанный запрос
            if result["query_type"] == "other" and processed.get("message") == "Извините, я не могу ответить на ваш вопрос":
                result["message"] = processed["message"]
                result["success"] = True
                return result
            
            # 5. Выполнение действий в зависимости от типа запроса
            if result["query_type"] == "recommendation":
                recommendations = self._handle_recommendation(processed)
                result["data"] = recommendations
                result["message"] = processed.get("message", "")
                
            elif result["query_type"] == "search":  # Убрали general
                result["data"] = processed.get("filtered_books")
                result["message"] = processed.get("message", "")
                
            elif result["query_type"] == "comparison":
                comparison_result = self._handle_comparison(processed)
                result["data"] = comparison_result
                result["message"] = processed.get("message", "")
                
            elif result["query_type"] == "step_back":
                result["data"] = processed.get("filtered_books")
                result["message"] = processed.get("message", "")
                # Показываем рекомендации на основе восстановленного состояния
                if processed.get("liked_indices"):
                    rec_result = self._handle_recommendation(processed)
                    if rec_result and rec_result.get("recommendations"):
                        result["data"] = rec_result
            
            elif result["query_type"] == "reset":
                result["data"] = processed.get("filtered_books")
                result["message"] = processed.get("message", "")
            
            else:
                result["message"] = "Извините, я не могу ответить на ваш вопрос"
            
            result["success"] = True
            
        except Exception as e:
            result["message"] = f"Ошибка обработки запроса: {str(e)}"
        
        return result
    
    def _handle_recommendation(self, processed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка запроса на рекомендации с учетом текущего состояния"""
        liked_indices = processed_query.get("liked_indices", [])
        disliked_indices = processed_query.get("disliked_indices", [])
        filtered_books = processed_query.get("filtered_books")
        
        if not liked_indices:
            return {
                "recommendations": [],
                "count": 0,
                "message": "Нет информации о понравившихся книгах для рекомендаций"
            }
        
        if filtered_books is None or len(filtered_books) == 0:
            return {
                "recommendations": [],
                "count": 0,
                "message": "Нет книг для рекомендаций после применения фильтров"
            }
        
        # Обновляем метрики для текущих отфильтрованных книг
        self.query_processor.metrics_filtered = BookDistanceMetrics(filtered_books)
        self.recommender.metrics_filtered = self.query_processor.metrics_filtered
        
        # Получаем рекомендации на основе текущего состояния
        recommendations = self.recommender.recommend_based_on_likes(
            liked_indices=liked_indices,
            disliked_indices=disliked_indices,
            n_recommendations=Config.DEFAULT_N_RECOMMENDATIONS,
            strategy=Config.DEFAULT_STRATEGY,
            penalty_factor=Config.DEFAULT_PENALTY_FACTOR
        )
        
        # Форматируем результат
        formatted_recs = []
        for book_idx, similarity in recommendations:
            book_info = self.query_processor.get_book_info(book_idx, is_filtered=True)
            if book_info:
                formatted_recs.append({
                    "book": book_info,
                    "similarity": float(similarity)
                })
        
        # Получаем информацию о лайках/дизлайках
        liked_books = []
        disliked_books = []
        
        for idx in liked_indices:
            book_info = self.query_processor.get_book_info(idx, is_filtered=False)
            if book_info:
                liked_books.append(book_info)
        
        for idx in disliked_indices:
            book_info = self.query_processor.get_book_info(idx, is_filtered=False)
            if book_info:
                disliked_books.append(book_info)
        
        return {
            "recommendations": formatted_recs,
            "count": len(formatted_recs),
            "liked_books": liked_books,
            "disliked_books": disliked_books,
            "filtered_books_count": len(filtered_books)
        }
    
    def _handle_comparison(self, processed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка запроса на сравнение"""
        comparison_books = processed_query.get("comparison_books", [])
        
        if len(comparison_books) < 2:
            return {"error": "Недостаточно книг для сравнения"}
        
        # Находим индексы книг в полном датасете
        book1_idx = comparison_books[0].name
        book2_idx = comparison_books[1].name
        
        # Сравниваем книги
        comparison_result = self.recommender.compare_books(book1_idx, book2_idx)
        
        return comparison_result
    
    def _display_current_state(self, state_info: Dict[str, Any]):
        """Отображение текущего состояния системы"""
        print(f"\n📊 ТЕКУЩЕЕ СОСТОЯНИЕ:")
        print(f"  Книг в фильтре: {state_info['books_count']}")
        print(f"  История: шаг {state_info['history']['current_step']} из {state_info['history']['max_steps']}")
        
        # Показываем ВСЕ лайки
        likes = state_info['preferences'].get('likes', [])
        if likes:
            print(f"\n👍 ПОНРАВИЛИСЬ ({len(likes)}):")
            for i, book in enumerate(likes, 1):
                print(f"  {i}. {book}")
        else:
            print(f"\n👍 ПОНРАВИЛИСЬ: нет")
        
        # Показываем ВСЕ дизлайки
        dislikes = state_info['preferences'].get('dislikes', [])
        if dislikes:
            print(f"\n👎 НЕ ПОНРАВИЛИСЬ ({len(dislikes)}):")
            for i, book in enumerate(dislikes, 1):
                print(f"  {i}. {book}")
        else:
            print(f"\n👎 НЕ ПОНРАВИЛИСЬ: нет")
        
        # Показываем ВСЕ активные фильтры
        active_filters = state_info.get('active_filters', {})
        if active_filters:
            print(f"\n🔍 АКТИВНЫЕ ФИЛЬТРЫ:")
            for key, value in active_filters.items():
                if isinstance(value, list):
                    print(f"  • {key}: {', '.join(value)}")
                else:
                    print(f"  • {key}: {value}")
        else:
            print(f"\n🔍 АКТИВНЫЕ ФИЛЬТРЫ: нет")
    
    def interactive_mode(self):
        """Интерактивный режим работы с поддержкой истории"""
        if not self.initialized:
            print("❌ Система не инициализирована")
            return
        
        print("\n" + "=" * 60)
        print("💬 ИНТЕРАКТИВНЫЙ РЕЖИМ С ИСТОРИЕЙ")
        print("=" * 60)
        print("Доступные команды:")
        print("  • 'назад' - вернуться на шаг назад (step_back: -1)")
        print("  • 'начать сначала' - начать с чистого листа (step_back: 1)")
        print("  • 'выход' - завершить работу")
        print("\nПримеры запросов:")
        print("  • 'Мне нравится Гарри Поттер'")
        print("  • 'Не нравится Война и мир'")
        print("  • 'Книги после 2020 года'")
        print("  • 'Короткие книги про любовь'")
        print("  • 'Очень длинная книга'")
        print("  • 'Сравни Войну и мир и Анну Каренину'")
        print("  • 'Рекомендуй что-то похожее'")
        print("=" * 60)
        
        while True:
            try:
                # Получаем запрос
                print("\n" + "-" * 40)
                query = input("🤔 Ваш запрос: ").strip()
                
                if query.lower() in ['выход', 'exit', 'quit', 'q']:
                    print("👋 До свидания!")
                    break
                
                if not query:
                    continue
                
                # Обрабатываем запрос
                result = self.process_user_query(query)
                
                # ВЫВОДИМ ТЕКУЩЕЕ СОСТОЯНИЕ ПЕРЕД РЕЗУЛЬТАТОМ
                if result.get("state_info"):
                    self._display_current_state(result["state_info"])
                
                # Выводим результат
                self._display_result(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 До свидания!")
                break
            except Exception as e:
                print(f"❌ Ошибка: {e}")
    
    def _display_result(self, result: Dict[str, Any]):
        """Отображение результата обработки запроса"""
        if not result.get("success"):
            print(f"\n❌ {result.get('message', 'Ошибка обработки запроса')}")
            return
        
        query_type = result.get("query_type", "")
        data = result.get("data")
        
        # Выводим сообщение о результате
        if result.get("message"):
            print(f"\n📝 {result.get('message')}")
        
        # Если запрос не распознан
        if query_type == "other" and result.get("message") == "Извините, я не могу ответить на ваш вопрос":
            print(f"\n⚠️  {result.get('message')}")
            return
        
        # Отображаем данные в зависимости от типа запроса
        if query_type == "recommendation" and data:
            recommendations = data.get("recommendations", [])
            
            if recommendations:
                print(f"\n🎯 РЕКОМЕНДАЦИИ ({len(recommendations)}):")
                
                for i, rec in enumerate(recommendations, 1):
                    book = rec["book"]
                    similarity = rec["similarity"]
                    print(f"\n{i}. {book['title']} - {book['author']}")
                    print(f"   Жанр: {book['genre']}, Год: {book['year']}, Страниц: {book['pages']}")
                    print(f"   Схожесть: {similarity:.3f}")
            else:
                print("\n❌ Не найдено подходящих рекомендаций")
        
        elif query_type == "search" and data is not None:
            print(f"\n🔍 РЕЗУЛЬТАТЫ ПОИСКА ({len(data)} книг):")
            if len(data) > 0:
                for i, (_, book) in enumerate(data.head(5).iterrows(), 1):
                    print(f"{i}. {book['title']} - {book['author']} ({book['genre']}, {book['year']} г.)")
                if len(data) > 5:
                    print(f"... и еще {len(data) - 5} книг")
        
        elif query_type == "comparison" and data:
            print(f"\n📊 СРАВНЕНИЕ КНИГ:")
            print(f"Схожесть: {data.get('similarity', 0):.3f}")
            
            book1 = data.get('book1', {})
            book2 = data.get('book2', {})
            
            print(f"\n📖 Книга 1: {book1.get('title', '')} - {book1.get('author', '')}")
            print(f"   Жанр: {book1.get('genre', '')}, Год: {book1.get('year', '')}, Страниц: {book1.get('pages', '')}")
            
            print(f"\n📖 Книга 2: {book2.get('title', '')} - {book2.get('author', '')}")
            print(f"   Жанр: {book2.get('genre', '')}, Год: {book2.get('year', '')}, Страниц: {book2.get('pages', '')}")
            
            differences = data.get('differences', [])
            if differences:
                print(f"\n⚠️  РАЗЛИЧИЯ:")
                for diff in differences:
                    print(f"   • {diff}")
            
            common = data.get('common_features', [])
            if common:
                print(f"\n✅ ОБЩИЕ ЧЕРТЫ:")
                for feature in common:
                    print(f"   • {feature}")


def main():
    """Главная функция"""
    system = BookRecommendationSystem()
    
    if not system.initialize():
        print("❌ Не удалось инициализировать систему")
        sys.exit(1)
    
    # Запуск интерактивного режима
    system.interactive_mode()


if __name__ == "__main__":
    main()