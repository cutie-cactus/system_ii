"""
Главный модуль системы рекомендаций книг
"""
import sys
from typing import Dict, Any
from config import Config
from data_loader import BookDataLoader
from neural_parser import NeuralBookParser
from query_processor import QueryProcessor
from book_recommender import BookRecommender

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
            print("📚 СИСТЕМА РЕКОМЕНДАЦИЙ КНИГ")
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
            
            # 3. Инициализация обработчика запросов
            print("\n⚙️ ИНИЦИАЛИЗАЦИЯ ОБРАБОТЧИКА...")
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
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"❌ Ошибка инициализации системы: {e}")
            return False
    
    def process_user_query(self, query: str) -> Dict[str, Any]:
        """
        Обработка пользовательского запроса
        
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
            "data": None
        }
        
        try:
            # 1. Парсинг запроса нейросетью
            parsed_query = self.neural_parser.parse_query(query)
            
            if not parsed_query or not parsed_query.get('question_type'):
                result["message"] = "Не удалось распознать запрос"
                return result
            
            result["query_type"] = parsed_query.get('question_type', '')
            
            # 2. Обработка распарсенного запроса
            processed = self.query_processor.process_query(parsed_query)
            
            # 3. Выполнение действий в зависимости от типа запроса
            if result["query_type"] == "recommendation":
                recommendations = self._handle_recommendation(processed, parsed_query)
                result["data"] = recommendations
                result["message"] = processed.get("message", "")
                
            elif result["query_type"] == "search":
                result["data"] = processed.get("filtered_books")
                result["message"] = processed.get("message", "")
                
            elif result["query_type"] == "comparison":
                comparison_result = self._handle_comparison(processed)
                result["data"] = comparison_result
                result["message"] = processed.get("message", "")
                
            elif result["query_type"] == "general":
                result["data"] = processed.get("filtered_books")
                result["message"] = processed.get("message", "")
                
            else:
                result["message"] = f"Тип запроса '{result['query_type']}' пока не поддерживается"
            
            result["success"] = True
            
        except Exception as e:
            result["message"] = f"Ошибка обработки запроса: {str(e)}"
        
        return result
    
    def _handle_recommendation(self, processed_query: Dict[str, Any], parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка запроса на рекомендации"""
        liked_indices = processed_query.get("liked_indices", [])
        disliked_indices = processed_query.get("disliked_indices", [])
        
        # Получаем рекомендации
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
        
        return {
            "recommendations": formatted_recs,
            "count": len(formatted_recs),
            "liked_books": [self.query_processor.get_book_info(idx) for idx in liked_indices if idx is not None],
            "disliked_books": [self.query_processor.get_book_info(idx) for idx in disliked_indices if idx is not None]
        }
    
    def _handle_comparison(self, processed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка запроса на сравнение"""
        comparison_books = processed_query.get("comparison_books", [])
        
        if len(comparison_books) < 2:
            return {"error": "Недостаточно книг для сравнения"}
        
        # Находим индексы книг
        book1_idx = comparison_books[0].name
        book2_idx = comparison_books[1].name
        
        # Сравниваем книги
        comparison_result = self.recommender.compare_books(book1_idx, book2_idx)
        
        return comparison_result
    
    def interactive_mode(self):
        """Интерактивный режим работы"""
        if not self.initialized:
            print("❌ Система не инициализирована")
            return
        
        print("\n" + "=" * 60)
        print("💬 ИНТЕРАКТИВНЫЙ РЕЖИМ")
        print("=" * 60)
        print("Введите ваш запрос или 'выход' для завершения")
        
        while True:
            try:
                print("\n" + "-" * 40)
                query = input("🤔 Ваш запрос: ").strip()
                
                if query.lower() in ['выход', 'exit', 'quit', 'q']:
                    print("👋 До свидания!")
                    break
                
                if not query:
                    continue
                
                # Обрабатываем запрос
                result = self.process_user_query(query)
                
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
            print(f"❌ {result.get('message', 'Ошибка обработки запроса')}")
            return
        
        query_type = result.get("query_type", "")
        data = result.get("data")
        
        print(f"\n📋 Тип запроса: {query_type}")
        print(f"📝 Результат: {result.get('message', '')}")
        
        if query_type == "recommendation" and data:
            recommendations = data.get("recommendations", [])
            print(f"\n🎯 Найдено рекомендаций: {len(recommendations)}")
            
            if recommendations:
                print("\n📚 РЕКОМЕНДАЦИИ:")
                for i, rec in enumerate(recommendations, 1):
                    book = rec["book"]
                    similarity = rec["similarity"]
                    print(f"\n{i}. {book['title']} - {book['author']}")
                    print(f"   Жанр: {book['genre']}, Год: {book['year']}, Страниц: {book['pages']}")
                    print(f"   Схожесть: {similarity:.3f}")
        
        elif query_type == "search" and data is not None:
            print(f"\n🔍 Найдено книг: {len(data)}")
            if len(data) > 0:
                print("\n📚 РЕЗУЛЬТАТЫ ПОИСКА:")
                for i, (_, book) in enumerate(data.head(10).iterrows(), 1):
                    print(f"{i}. {book['title']} - {book['author']} ({book['genre']}, {book['year']} г.)")
                if len(data) > 10:
                    print(f"... и еще {len(data) - 10} книг")
        
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