import json
import sys
from typing import Dict, Any

class BookRecommendationInterface:
    """
    Управляющая программа с интерфейсом для книжной рекомендательной системы
    """
    def __init__(self, neural_parser, recommendation_system):
        """
        Инициализация интерфейса
        
        Args:
            neural_parser: экземпляр NeuralBookParser
            recommendation_system: экземпляр BookRecommendationSystem
        """
        self.neural_parser = neural_parser
        self.recommendation_system = recommendation_system
        self.is_running = False
    
    def start(self):
        """
        Запуск интерфейса
        """
        print("=" * 60)
        print("📚 КНИЖНАЯ РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА")
        print("=" * 60)
        print()
        
        # Инициализация нейросети
        print("🧠 Инициализация нейросети...")
        if not self.neural_parser.initialize():
            print("❌ Не удалось инициализировать нейросеть. Проверьте подключение.")
            return
        
        print("✅ Нейросеть готова к работе!")
        print()
        
        self.is_running = True
        self._main_loop()
    
    def _main_loop(self):
        """
        Главный цикл интерфейса
        """
        while self.is_running:
            try:
                # Получаем запрос от пользователя
                user_input = self._get_user_input()
                
                if user_input.lower() in ['выход', 'exit', 'quit', 'q']:
                    self._handle_exit()
                    continue
                
                if user_input.lower() in ['история', 'history', 'h']:
                    self._show_history()
                    continue
                
                if user_input.lower() in ['статус', 'status', 's']:
                    self._show_status()
                    continue
                
                if user_input.lower() in ['фильтры', 'filters', 'f']:
                    self._show_current_filters()
                    continue
                
                if user_input.lower() in ['очистить', 'clear', 'c']:
                    self._clear_filters()
                    continue
                
                # Обрабатываем запрос
                self._process_user_query(user_input)
                
            except KeyboardInterrupt:
                self._handle_exit()
            except Exception as e:
                print(f"❌ Ошибка: {e}")
                print("Попробуйте еще раз или введите 'выход' для завершения.")
    
    def _get_user_input(self) -> str:
        """
        Получение ввода от пользователя
        
        Returns:
            Введенная строка
        """
        print("\n" + "─" * 40)
        print("💬 Введите ваш запрос (или 'помощь' для списка команд):")
        user_input = input("> ").strip()
        
        if not user_input:
            return ""
        
        # Обработка специальных команд помощи
        if user_input.lower() in ['помощь', 'help', '?']:
            self._show_help()
            return ""
        
        return user_input
    
    def _process_user_query(self, user_query: str):
        """
        Обработка пользовательского запроса
        
        Args:
            user_query: запрос пользователя
        """
        print(f"\n🔍 Обрабатываю запрос: '{user_query}'")
        
        # 1. Парсим запрос с помощью нейросети
        print("🤖 Анализирую запрос нейросетью...")
        parsed_query = self.neural_parser.parse_query(user_query)
        
        # Показываем результат парсинга
        self._show_parsed_query(parsed_query)
        
        # 2. Передаем в рекомендательную систему - используем правильное название метода
        print("📊 Обрабатываю запрос в рекомендательной системе...")
        result = self.recommendation_system.process_query(parsed_query)  # ИСПРАВЛЕНО: process_query вместо process_parsed_query
        
        # 3. Выводим результат пользователю
        self._show_result(result)
    
    def _show_parsed_query(self, parsed_query: Dict[str, Any]):
        """
        Показывает распарсенный запрос
        
        Args:
            parsed_query: результат парсинга нейросети
        """
        print("📋 Распарсенный запрос:")
        
        question_type = parsed_query.get("question_type", "неизвестно")
        if question_type == "неизвестно":
            question_type = parsed_query.get("class", "неизвестно")

        print(f"   Тип вопроса: {question_type}")
        
        # Показываем фильтры если они есть
        filters = parsed_query.get("filter", {})
        non_empty_filters = {k: v for k, v in filters.items() if v and (v != [] and v != "" and v != {})}
        if non_empty_filters:
            print("   Фильтры:")
            for key, value in non_empty_filters.items():
                print(f"     - {key}: {value}")
        
        # Показываем сравнение если есть
        compare = parsed_query.get("compare", {})
        if compare.get("count_books"):
            print(f"   Сравнение: {compare.get('count_books')} книг")
            for i in range(1, int(compare.get("count_books", 0)) + 1):
                title = compare.get(f"title{i}")
                author = compare.get(f"author{i}")
                if title or author:
                    print(f"     Книга {i}: {title} - {author}")
        
        # Показываем feedback если есть
        feedback = parsed_query.get("feedback", {})
        if feedback.get("likes"):
            print(f"   👍 Нравятся: {', '.join(feedback['likes'])}")
        if feedback.get("dislikes"):
            print(f"   👎 Не нравятся: {', '.join(feedback['dislikes'])}")
        
        print()
    
    def _show_result(self, result: Dict[str, Any]):
        """
        Показывает результат обработки
        
        Args:
            result: результат от рекомендательной системы
        """
        print("\n🎯 РЕЗУЛЬТАТ:")
        print("─" * 40)
        
        if not result.get("success", False):
            print("❌ " + result.get("message", "Произошла ошибка"))
            return
        
        # Основное сообщение
        print("✅ " + result.get("message", ""))
        
        # Информация о книгах
        books = result.get("books", [])
        total_count = result.get("total_count", 0)
        
        if total_count > 0:
            print(f"\n📚 Найдено книг: {total_count}")
            
            for i, book in enumerate(books[:10], 1):  # Ограничиваем 10 книгами
                print(f"\n{i}. 📖 {book.get('title', 'Без названия')}")
                print(f"   ✍️  Автор: {book.get('author', 'Неизвестен')}")
                print(f"   📅 Год: {book.get('year', 'Неизвестен')}")
                print(f"   🎭 Жанр: {book.get('genre', 'Неизвестен')}")
                print(f"   📄 Страниц: {book.get('pages', 'Неизвестно')}")
                print(f"   👶 Возраст: {book.get('age_restriction', 'Не указано')}")
                
                if book.get('has_illustrations'):
                    print(f"   🖼️  С иллюстрациями")
            
            if total_count > 10:
                print(f"\n... и еще {total_count - 10} книг")
        else:
            print("\n📭 Книги не найдены")
        
        # Текущие фильтры
        current_filters = result.get("current_filters", {})
        if current_filters and any(current_filters.values()):
            print(f"\n🔍 Активные фильтры:")
            for key, value in current_filters.items():
                if value and (value != [] and value != "" and value != {}):
                    print(f"   - {key}: {value}")
        
        print("─" * 40)
    
    def _show_help(self):
        """
        Показывает справку по командам
        """
        print("\n📖 СПРАВКА ПО КОМАНДАМ:")
        print("─" * 40)
        print("Основные команды:")
        print("  помощь, help, ?     - показать эту справку")
        print("  выход, exit, quit   - выйти из программы")
        print("  история, history     - показать историю запросов")
        print("  статус, status       - показать статус системы")
        print("  фильтры, filters     - показать текущие фильтры")
        print("  очистить, clear      - очистить фильтры и историю")
        print()
        print("Примеры запросов:")
        print("  • Найди книги Стивена Кинга")
        print("  • Посоветуй что-то похожее на Гарри Поттера")
        print("  • Сравни 'Война и мир' и 'Анна Каренина'")
        print("  • Сколько страниц в книге 'Властелин колец'?")
        print("  • Кто издатель книги '1984'?")
        print("  • Книги в жанре фэнтези для детей")
        print("  • Вернись к предыдущему запросу")
        print("─" * 40)
    
    def _show_history(self):
        """
        Показывает историю запросов
        """
        history = self.recommendation_system.get_history()
        
        print("\n🕐 ИСТОРИЯ ЗАПРОСОВ:")
        print("─" * 40)
        
        if not history:
            print("История пуста")
            return
        
        for i, entry in enumerate(history, 1):
            timestamp = entry.get("timestamp", "")[:19]  # Берем только время
            query_type = entry.get("query", {}).get("question_type", "неизвестно")
            result_count = entry.get("result", {}).get("total_count", 0)
            
            print(f"{i}. [{timestamp}] {query_type} - {result_count} книг")
        
        print("─" * 40)
    
    def _show_status(self):
        """
        Показывает статус системы
        """
        neural_status = self.neural_parser.get_status()
        current_filters = self.recommendation_system.get_current_filters()
        
        print("\n📊 СТАТУС СИСТЕМЫ:")
        print("─" * 40)
        print(f"🧠 Нейросеть: {'✅ Готова' if neural_status['initialized'] else '❌ Не готова'}")
        print(f"📚 База данных: {len(self.recommendation_system.df)} книг")
        print(f"📖 История запросов: {len(self.recommendation_system.query_history)}")
        
        active_filters = sum(1 for v in current_filters.values() if v and (v != [] and v != "" and v != {}))
        print(f"🔍 Активных фильтров: {active_filters}")
        
        if active_filters > 0:
            print("\nТекущие фильтры:")
            for key, value in current_filters.items():
                if value and (value != [] and value != "" and value != {}):
                    print(f"  - {key}: {value}")
        
        print("─" * 40)
    
    def _show_current_filters(self):
        """
        Показывает текущие фильтры
        """
        current_filters = self.recommendation_system.get_current_filters()
        
        print("\n🔍 ТЕКУЩИЕ ФИЛЬТРЫ:")
        print("─" * 40)
        
        has_filters = False
        for key, value in current_filters.items():
            if value and (value != [] and value != "" and value != {}):
                print(f"  {key}: {value}")
                has_filters = True
        
        if not has_filters:
            print("Фильтры не установлены")
        
        print("─" * 40)
    
    def _clear_filters(self):
        """
        Очищает фильтры и историю
        """
        # Создаем запрос на очистку
        clear_query = {
            "question_type": "step_back",
            "filter": {},
            "compare": {},
            "feedback": {},
            "num_question": "",
            "step_back": "1"
        }
        
        # Обрабатываем запрос очистки
        self.recommendation_system.process_query(clear_query)  # ИСПРАВЛЕНО: process_query вместо process_parsed_query
        print("✅ Фильтры и история очищены")
    
    def _handle_exit(self):
        """
        Обработка выхода из программы
        """
        print("\n👋 До свидания! Спасибо за использование книжной рекомендательной системы!")
        self.is_running = False


# Главная функция для запуска программы
def main():
    """
    Главная функция запуска интерфейса
    """
    try:
        # Инициализация компонентов
        from llamaParser import NeuralBookParser
        from funcDialog import BookRecommendationSystem
        # Создаем экземпляры
        neural_parser = NeuralBookParser()
        recommendation_system = BookRecommendationSystem()
        
        # Создаем и запускаем интерфейс
        interface = BookRecommendationInterface(neural_parser, recommendation_system)
        interface.start()
        
    except ImportError as e:
        print(f"❌ Ошибка импорта модулей: {e}")
        print("Убедитесь, что модули NeuralBookParser и BookRecommendationSystem доступны")
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")


if __name__ == "__main__":
    main()
