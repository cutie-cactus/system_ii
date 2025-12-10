import requests
import json
import re
import time
from typing import Dict, Any, Optional

class NeuralBookParser:
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Инициализация парсера с нейросетью Llama 3.1
        
        Args:
            base_url: URL локально запущенной нейросети Ollama
        """
        self.base_url = base_url
        self.completion_url = f"{base_url}/api/chat"
        self.system_prompt = ""
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """
        Инициализация нейросети: проверка доступности и загрузка промпта
        
        Returns:
            True если инициализация успешна
        """
        print("🔍 Проверяю доступность нейросети...")
        
        # Проверяем доступность нейросети
        if not self._test_connection():
            print("❌ Нейросеть недоступна")
            return False
        
        print("✅ Нейросеть доступна")
        
        # Загружаем системный промпт
        print("📖 Загружаю системный промпт...")
        self.system_prompt = self._load_system_prompt()
        
        if not self.system_prompt:
            print("❌ Не удалось загрузить системный промпт")
            return False
        
        print("✅ Системный промпт загружен")
        
        # Отправляем промпт для инициализации нейросети
        print("🚀 Инициализирую нейросеть системным промптом...")
        initialization_success = self._initialize_neural_network()
        
        if initialization_success:
            print("✅ Нейросеть успешно инициализирована")
            self.is_initialized = True
            return True
        else:
            print("❌ Ошибка инициализации нейросети")
            return False
    
    def _test_connection(self) -> bool:
        """
        Тестирование подключения к нейросети
        
        Returns:
            True если подключение успешно
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Ошибка подключения: {e}")
            return False
    
    def _load_system_prompt(self) -> str:
        """
        Загрузка системного промпта из файла
        
        Returns:
            Содержимое системного промпта
        """
        try:
            with open('mainPromtOnStart.md', 'r', encoding='utf-8') as file:
                content = file.read().strip()
                print(f"📄 Загружен промпт длиной {len(content)} символов")
                return content
        except FileNotFoundError:
            print("⚠️ Файл mainPromtOnStart.md не найден")
            return ""
        except Exception as e:
            print(f"⚠️ Ошибка загрузки промпта: {e}")
            return ""
    
    def _initialize_neural_network(self) -> bool:
        """
        Инициализация нейросети - теперь только проверка работы
        """
        try:
            # Простой тестовый запрос для проверки работы нейросети
            test_payload = {
                "model": "llama3.1:8b-instruct-q4_0",
                "messages": [
                    {"role": "user", "content": "Ответь 'READY' для подтверждения работы"}
                ],
                "stream": False
            }
            
            response = requests.post(
                self.completion_url,
                json=test_payload,
                headers={"Content-Type": "application/json"},
                timeout=50  # Возвращаем нормальный таймаут для теста
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["message"]["content"].strip()
                print(f"🤖 Ответ нейросети при тесте: {content}")
                return True
            else:
                print(f"❌ Ошибка теста: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка при тесте нейросети: {e}")
            return False

    def parse_query(self, user_query: str) -> Dict[str, Any]:
        """
        Парсинг пользовательского запроса с помощью нейросети
        """
        if not self.is_initialized:
            print("❌ Нейросеть не инициализирована. Вызовите initialize() сначала.")
            return self._get_empty_template()
        
        print(f"🤖 Обрабатываю запрос: '{user_query}'")
        
        try:
            # В КАЖДОМ запросе отправляем системный промпт + пользовательский запрос
            payload = {
                "model": "llama3.1:8b-instruct-q4_0",
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_query}
                ],
                "stream": False
            }
            
            start_time = time.time()
            response = requests.post(
                self.completion_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=180
            )
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result["message"]["content"].strip()
                
                print(f"✅ Ответ получен за {processing_time:.2f} сек")
                
                # Пытаемся извлечь JSON из ответа
                json_data = self._extract_json_from_response(content)
                print(json_data)
                # ВСЕГДА нормализуем структуру (добавляем отсутствующие поля)
                normalized_data = self._normalize_json_structure(json_data)
                print("✅ JSON нормализован")
                return normalized_data
                    
            else:
                print(f"❌ Ошибка запроса к нейросети: {response.status_code}")
                return self._get_empty_template()
                
        except requests.exceptions.Timeout:
            print("❌ Таймаут при запросе к нейросети")
            return self._get_empty_template()
        except requests.exceptions.ConnectionError:
            print("❌ Не удалось подключиться к нейросети")
            return self._get_empty_template()
        except Exception as e:
            print(f"❌ Ошибка при работе с нейросетью: {e}")
            return self._get_empty_template()

    def _normalize_json_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Нормализация структуры JSON - добавление отсутствующих полей
        """
        # Берем пустой шаблон
        normalized = self._get_empty_template()
        
        # Если data не словарь или пустой, возвращаем шаблон
        if not isinstance(data, dict) or not data:
            return normalized
        
        # ОБНОВЛЕНИЕ: Проверяем разные возможные названия для типа вопроса
        question_type = None
        for possible_key in ['question_type', 'class', 'type', 'questionType']:
            if possible_key in data and data[possible_key]:
                question_type = data[possible_key]
                break
        
        if question_type:
            # Сопоставляем с правильными названиями категорий
            type_mapping = {
                "recommendation": "recommendation",
                "search": "search", 
                "find": "search",
                "compare": "comparison",
                "comparison": "comparison",
                "general_question": "general",
                "general": "general",
                "question": "general",
                "step_back": "step_back",
                "back": "step_back",
                "return": "step_back",
                "other": "other",
                "hello": "other",
                "help": "other",
                "thanks": "other",
                "thank you": "other",
                "bye": "other",
                "exit": "other"
            }
            normalized["question_type"] = type_mapping.get(question_type.lower(), question_type)
        
        # ОБНОВЛЕНИЕ: Обрабатываем фильтры - проверяем разные структуры
        filter_data = data.get("filter", {})
        if not filter_data:
            # Проверяем, нет ли фильтров в корне объекта
            filter_fields = ["author", "publisher", "year_from", "year_to", "language", 
                        "age_restriction", "genre", "pages_from", "pages_to", "has_illustrations"]
            root_filter = {}
            for field in filter_fields:
                if field in data:
                    root_filter[field] = data[field]
            if root_filter:
                filter_data = root_filter
        
        if isinstance(filter_data, dict):
            for field in normalized["filter"]:
                if field in filter_data:
                    value = filter_data[field]
                    if value is not None and value != "":
                        # Для полей-списков преобразуем в список
                        if field in ["author", "publisher", "language", "age_restriction", "genre"]:
                            if isinstance(value, list):
                                normalized["filter"][field] = [str(item) for item in value if item]
                            elif isinstance(value, str) and value:
                                normalized["filter"][field] = [value]
                            elif value:
                                normalized["filter"][field] = [str(value)]
                        else:
                            # Для остальных полей просто преобразуем в строку
                            normalized["filter"][field] = str(value)
        
        # ОБНОВЛЕНИЕ: Обрабатываем сравнение
        compare_data = data.get("compare", {})
        if isinstance(compare_data, dict):
            for field in normalized["compare"]:
                if field in compare_data:
                    value = compare_data[field]
                    if value is not None and value != "":
                        normalized["compare"][field] = str(value)
        
        # ОБНОВЛЕНИЕ: Обрабатываем feedback - проверяем разные структуры
        feedback_data = data.get("feedback", {})
        if not feedback_data:
            # Проверяем альтернативные названия для feedback
            for possible_key in ['likes', 'dislikes', 'preferences', 'feedback']:
                if possible_key in data:
                    if possible_key == 'likes':
                        feedback_data = {"likes": data['likes']}
                    elif possible_key == 'dislikes':
                        feedback_data = {"dislikes": data['dislikes']}
                    elif isinstance(data[possible_key], dict):
                        feedback_data = data[possible_key]
                    break
        
        if isinstance(feedback_data, dict):
            for field in ["likes", "dislikes"]:
                if field in feedback_data:
                    value = feedback_data[field]
                    if value is not None:
                        if isinstance(value, list):
                            normalized["feedback"][field] = [str(item) for item in value if item]
                        elif isinstance(value, str) and value:
                            normalized["feedback"][field] = [value]
                        elif value:
                            normalized["feedback"][field] = [str(value)]
        
        # ОБНОВЛЕНИЕ: Обрабатываем опциональные поля
        for field in ["num_question", "step_back"]:
            if field in data:
                value = data[field]
                if value is not None and value != "":
                    normalized[field] = str(value)
        
        # ОБНОВЛЕНИЕ: Если тип вопроса все еще не определен, пытаемся определить по содержимому
        if not normalized["question_type"]:
            if normalized["feedback"].get("likes") or normalized["feedback"].get("dislikes"):
                normalized["question_type"] = "recommendation"
            elif normalized["compare"].get("count_books") and int(normalized["compare"].get("count_books", 0)) >= 2:
                normalized["question_type"] = "comparison"
            elif any(normalized["filter"].values()):
                normalized["question_type"] = "search"
            elif normalized.get("step_back"):
                normalized["question_type"] = "step_back"
            elif normalized.get("num_question"):
                normalized["question_type"] = "other"
        
        return normalized
    
    def _extract_json_from_response(self, content: str) -> Dict[str, Any]:
        """
        Извлечение JSON из ответа нейросети
        """
        try:
            # Удаляем возможные обертки кода
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Если нет обертки, ищем JSON-объект
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = content
            
            # # Чистим строку от лишних символов
            # json_str = json_str.strip()
            
            # # ОБНОВЛЕНИЕ: Пытаемся исправить распространенные ошибки в JSON
            # json_str = self._fix_common_json_errors(json_str)
            
            # Парсим JSON
            parsed_data = json.loads(json_str)
            return parsed_data
            
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка парсинга JSON: {e}")
            print(f"📝 Содержимое ответа нейросети: {content[:500]}")
            return self._get_empty_template()
        except Exception as e:
            print(f"❌ Неожиданная ошибка при извлечении JSON: {e}")
            return self._get_empty_template()

    def _fix_common_json_errors(self, json_str: str) -> str:
        """
        Исправление распространенных ошибок в JSON от нейросети
        """
        # Заменяем одиночные кавычки на двойные
        json_str = re.sub(r"(?<!\\)'", '"', json_str)
        
        # Убираем лишние запятые в конце объектов и массивов
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Исправляем незакрытые строки
        json_str = re.sub(r':\s*([^"\s][^,}\]]*?)\s*([,}\]])', r': "\1"\2', json_str)
        
        return json_str
    
    def _validate_json_structure(self, data: Dict[str, Any]) -> bool:
        """
        Проверка корректности структуры JSON в зависимости от типа вопроса
        
        Args:
            data: данные для проверки
            
        Returns:
            True если структура корректна для данного типа вопроса
        """
        try:
            # Проверяем, что это словарь
            if not isinstance(data, dict):
                return False
            
            # Проверяем наличие question_type
            if "question_type" not in data:
                return False
            
            question_type = data["question_type"]
            
            # Нормализуем название типа вопроса
            question_type = self._normalize_question_type(question_type)
            
            # Проверяем базовые типы для существующих полей
            if "filter" in data and not isinstance(data["filter"], dict):
                return False
            if "compare" in data and not isinstance(data["compare"], dict):
                return False
            if "feedback" in data and not isinstance(data["feedback"], dict):
                return False
            
            # В зависимости от типа вопроса проверяем только нужные поля
            if question_type == "recommendation":
                # Для рекомендаций проверяем feedback если он есть
                if "feedback" in data:
                    if not isinstance(data["feedback"].get("likes", []), list):
                        return False
                    if not isinstance(data["feedback"].get("dislikes", []), list):
                        return False
            
            elif question_type == "search":
                # Для поиска проверяем filter если он есть
                if "filter" in data:
                    for field in ["author", "publisher", "language", "age_restriction", "genre"]:
                        if field in data["filter"] and not isinstance(data["filter"][field], list):
                            # Если поле есть, но не список, пытаемся преобразовать
                            if not isinstance(data["filter"][field], (str, int, float)):
                                return False
            
            elif question_type == "compare":
                # Для сравнения проверяем compare
                if "compare" not in data:
                    return False
            
            elif question_type == "general_question":
                # Для общих вопросов должен быть хотя бы один способ идентифицировать книгу
                has_identifiers = False
                if "filter" in data:
                    for field in ["author", "publisher", "language", "age_restriction", "genre"]:
                        if data["filter"].get(field):
                            has_identifiers = True
                            break
                if not has_identifiers:
                    # Проверяем, есть ли информация в compare для идентификации
                    if "compare" in data:
                        for i in range(1, 3):
                            if data["compare"].get(f"title{i}"):
                                has_identifiers = True
                                break
                    if not has_identifiers:
                        return False
            
            elif question_type == "step_back":
                # Для шага назад должно быть значение step_back
                step_back = data.get("step_back", "")
                if step_back not in ["1", "-1"]:
                    return False
            
            elif question_type == "other":
                # Для других вопросов должен быть num_question
                num_question = data.get("num_question", "")
                if not num_question:
                    return False
            
            else:
                # Неизвестный тип вопроса
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка валидации JSON структуры: {e}")
            return False

    def _normalize_question_type(self, question_type: str) -> str:
        """
        Нормализация типа вопроса для единообразия
        
        Args:
            question_type: исходный тип вопроса
            
        Returns:
            Нормализованный тип вопроса
        """
        # Приводим к нижнему регистру и убираем пробелы
        question_type = str(question_type).lower().strip()
        
        type_mapping = {
            "recommend": "recommendation",
            "search": "search",
            "find": "search",
            "lookup": "search",
            "compare": "compare", 
            "comparison": "compare",
            "general": "general_question",
            "general_question": "general_question",
            "question": "general_question",
            "step_back": "step_back",
            "back": "step_back",
            "return": "step_back",
            "other": "other",
            "hello": "other",
            "help": "other",
            "thanks": "other",
            "bye": "other"
        }
        
        return type_mapping.get(question_type, question_type)

    def _normalize_json_structure2(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Нормализация структуры JSON - добавление отсутствующих полей
        
        Args:
            data: данные от нейросети
            
        Returns:
            Нормализованный словарь с всеми необходимыми полями
        """
        # Берем пустой шаблон
        normalized = self._get_empty_template()
        
        # Обновляем question_type если есть (и нормализуем его)
        if "question_type" in data:
            normalized["question_type"] = self._normalize_question_type(data["question_type"])
        
        # Обновляем filter (если есть)
        if "filter" in data and isinstance(data["filter"], dict):
            for field in normalized["filter"]:
                if field in data["filter"]:
                    # Приводим к правильному типу
                    if field in ["author", "publisher", "language", "age_restriction", "genre"]:
                        # Эти поля должны быть списками
                        if isinstance(data["filter"][field], list):
                            normalized["filter"][field] = data["filter"][field]
                        elif data["filter"][field]:
                            # Если это строка или число, создаем список
                            normalized["filter"][field] = [str(data["filter"][field])]
                    else:
                        # Остальные поля как есть (преобразуем в строку если нужно)
                        value = data["filter"][field]
                        if value is not None:
                            normalized["filter"][field] = str(value)
        
        # Обновляем compare (если есть)
        if "compare" in data and isinstance(data["compare"], dict):
            for field in normalized["compare"]:
                if field in data["compare"]:
                    value = data["compare"][field]
                    if value is not None:
                        normalized["compare"][field] = str(value)
        
        # Обновляем feedback (если есть)
        if "feedback" in data and isinstance(data["feedback"], dict):
            for field in ["likes", "dislikes"]:
                if field in data["feedback"]:
                    if isinstance(data["feedback"][field], list):
                        normalized["feedback"][field] = data["feedback"][field]
                    elif data["feedback"][field]:
                        # Если это одиночное значение, создаем список
                        normalized["feedback"][field] = [str(data["feedback"][field])]
        
        # Обновляем опциональные поля
        if "num_question" in data:
            normalized["num_question"] = str(data["num_question"])
        if "step_back" in data:
            normalized["step_back"] = str(data["step_back"])
        
        return normalized
    
    def _get_empty_template(self) -> Dict[str, Any]:
        """
        Создание пустого шаблона ответа
        
        Returns:
            Пустой шаблон JSON
        """
        return {
            "question_type": "",
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
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Получение статуса нейросети
        
        Returns:
            Словарь со статусом
        """
        return {
            "initialized": self.is_initialized,
            "neural_network_available": self._test_connection(),
            "prompt_loaded": bool(self.system_prompt),
            "prompt_length": len(self.system_prompt) if self.system_prompt else 0
        }


# Пример использования
def main():
    # Инициализация нейросети
    neural_parser = NeuralBookParser()
    
    print("=" * 60)
    print("🧠 ИНИЦИАЛИЗАЦИЯ НЕЙРОСЕТИ")
    print("=" * 60)
    
    # Инициализируем нейросеть
    if neural_parser.initialize():
        print("✅ Нейросеть готова к работе!")
        
        # Показываем статус
        status = neural_parser.get_status()
        print(f"📊 Статус: {status}")
        
        # Тестовые запросы
        test_queries = [
            "Найди книги Стивена Кинга в жанре ужасы",
            "Найди книги Льва Тодстого",
            "Посоветуй что-то похожее на Гарри Поттера",
            "Сравни Войну и мир и Анну Каренину",
            "Хочу короткие книги"
        ]
        
        for query in test_queries:
            print(f"\n🎯 Запрос: {query}")
            result = neural_parser.parse_query(query)
            print(f"📋 Результат: {json.dumps(result, ensure_ascii=False)}")
            
    else:
        print("❌ Не удалось инициализировать нейросеть")


if __name__ == "__main__":
    main()

