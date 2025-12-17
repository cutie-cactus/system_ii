"""
Парсер запросов с использованием нейросети
"""
import requests
import json
import re
import time
from typing import Dict, Any, Optional
from config import Config

class NeuralBookParser:
    def __init__(self, base_url: str = Config.NEURAL_URL):
        """
        Инициализация парсера с нейросетью
        """
        self.base_url = base_url
        self.completion_url = f"{base_url}/api/chat"
        self.system_prompt = ""
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """
        Инициализация нейросети
        """
        print("🔍 Проверяю доступность нейросети...")
        
        if not self._test_connection():
            print("❌ Нейросеть недоступна")
            return False
        
        print("✅ Нейросеть доступна")
        print("📖 Загружаю системный промпт...")
        
        self.system_prompt = self._load_system_prompt()
        if not self.system_prompt:
            print("❌ Не удалось загрузить системный промпт")
            return False
        
        print("✅ Системный промпт загружен")
        print("🚀 Инициализирую нейросеть...")
        
        initialization_success = self._initialize_neural_network()
        
        if initialization_success:
            print("✅ Нейросеть успешно инициализирована")
            self.is_initialized = True
            return True
        else:
            print("❌ Ошибка инициализации нейросети")
            return False
    
    def _test_connection(self) -> bool:
        """Проверка подключения к нейросети"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Ошибка подключения: {e}")
            return False
    
    def _load_system_prompt(self) -> str:
        """Загрузка системного промпта"""
        try:
            with open(Config.PROMPT_PATH, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                
                # Добавляем информацию о командах в промпт
                enhanced_content = content + """

ДОПОЛНИТЕЛЬНЫЕ ИНСТРУКЦИИ ДЛЯ КОМАНД:

1. Если пользователь говорит "назад", "вернись назад", "отмени последнее", "шаг назад" - 
   используй question_type: "step_back"

2. Если пользователь говорит "заново", "начать сначала", "сбросить все", "очистить" - 
   используй question_type: "other" и num_question: "заново"

3. Для добавления предпочтений используй естественные фразы:
   - "Мне нравится [название книги]" → feedback: {"likes": ["название книги"]}
   - "Не нравится [название книги]" → feedback: {"dislikes": ["название книги"]}
   - "Еще мне нравится [название]" → добавляй в существующие likes

4. Для фильтров сохраняй все ранее указанные критерии плюс новые:
   - "Книги после 2020 года" → добавляй к существующим фильтрам
   - "И еще русские книги" → добавляй language: ["русский"] к существующим фильтрам

5. Все запросы на рекомендации без явного указания книг должны учитывать 
   накопленные предпочтения (likes/dislikes) и фильтры.
"""
                
                print(f"📄 Загружен промпт длиной {len(enhanced_content)} символов")
                return enhanced_content
        except FileNotFoundError:
            print(f"⚠️ Файл {Config.PROMPT_PATH} не найден")
            return ""
        except Exception as e:
            print(f"⚠️ Ошибка загрузки промпта: {e}")
            return ""
    
    def _initialize_neural_network(self) -> bool:
        """Инициализация нейросети"""
        try:
            test_payload = {
                "model": Config.NEURAL_MODEL,
                "messages": [
                    {"role": "user", "content": "Ответь 'READY' для подтверждения работы"}
                ],
                "stream": False
            }
            
            response = requests.post(
                self.completion_url,
                json=test_payload,
                headers={"Content-Type": "application/json"},
                timeout=150
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
        Парсинг пользовательского запроса
        """
        if not self.is_initialized:
            print("❌ Нейросеть не инициализирована")
            return self._get_empty_template()
        
        print(f"🤖 Обрабатываю запрос: '{user_query}'")
        
        try:
            payload = {
                "model": Config.NEURAL_MODEL,
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
                timeout=Config.NEURAL_TIMEOUT
            )
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result["message"]["content"].strip()
                
                print(f"✅ Ответ получен за {processing_time:.2f} сек")
                
                json_data = self._extract_json_from_response(content)
                normalized_data = self._normalize_json_structure(json_data)
                
                # Логируем результат для отладки
                print(f"📋 Распознанный тип: {normalized_data.get('question_type', 'неизвестно')}")
                if normalized_data.get('feedback', {}).get('likes'):
                    print(f"👍 Лайки: {normalized_data['feedback']['likes']}")
                if normalized_data.get('feedback', {}).get('dislikes'):
                    print(f"👎 Дизлайки: {normalized_data['feedback']['dislikes']}")
                
                return normalized_data
                    
            else:
                print(f"❌ Ошибка запроса к нейросети: {response.status_code}")
                return self._get_empty_template()
                
        except Exception as e:
            print(f"❌ Ошибка при работе с нейросетью: {e}")
            return self._get_empty_template()

    def _extract_json_from_response(self, content: str) -> Dict[str, Any]:
        """Извлечение JSON из ответа нейросети"""
        try:
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = content
            
            parsed_data = json.loads(json_str)
            return parsed_data
            
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка парсинга JSON: {e}")
            print(f"📝 Содержимое ответа нейросети: {content[:500]}")
            return self._get_empty_template()
        except Exception as e:
            print(f"❌ Неожиданная ошибка при извлечении JSON: {e}")
            return self._get_empty_template()

    def _normalize_json_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Нормализация структуры JSON"""
        normalized = self._get_empty_template()
        
        if not isinstance(data, dict) or not data:
            return normalized
        
        # Определение типа вопроса
        question_type = None
        for possible_key in ['question_type', 'class', 'type', 'questionType']:
            if possible_key in data and data[possible_key]:
                question_type = data[possible_key]
                break
        
        if question_type:
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
                "other": "other"
            }
            normalized["question_type"] = type_mapping.get(question_type.lower(), question_type)
        
        # Проверка командных фраз
        if 'step_back' in data and data['step_back']:
            normalized['question_type'] = 'step_back'
        elif 'num_question' in data and data['num_question'] == 'заново':
            normalized['question_type'] = 'other'
            normalized['num_question'] = 'заново'
        
        # Обработка фильтров
        filter_data = data.get("filter", {})
        if not filter_data:
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
                        if field in ["author", "publisher", "language", "age_restriction", "genre"]:
                            if isinstance(value, list):
                                normalized["filter"][field] = [str(item) for item in value if item]
                            elif isinstance(value, str) and value:
                                normalized["filter"][field] = [value]
                            elif value:
                                normalized["filter"][field] = [str(value)]
                        else:
                            normalized["filter"][field] = str(value)
        
        # Обработка сравнения
        compare_data = data.get("compare", {})
        if isinstance(compare_data, dict):
            for field in normalized["compare"]:
                if field in compare_data:
                    value = compare_data[field]
                    if value is not None and value != "":
                        normalized["compare"][field] = str(value)
        
        # Обработка feedback
        feedback_data = data.get("feedback", {})
        if not feedback_data:
            for possible_key in ['likes', 'dislikes', 'preferences']:
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
        
        # Обработка опциональных полей
        for field in ["num_question", "step_back"]:
            if field in data:
                value = data[field]
                if value is not None and value != "":
                    normalized[field] = str(value)
        
        # Автоматическое определение типа вопроса по содержимому
        if not normalized["question_type"]:
            if normalized["feedback"].get("likes") or normalized["feedback"].get("dislikes"):
                normalized["question_type"] = "recommendation"
            elif normalized["compare"].get("title1") and normalized["compare"].get("title2"):
                normalized["question_type"] = "comparison"
            elif any(normalized["filter"].values()):
                normalized["question_type"] = "search"
            elif normalized.get("step_back"):
                normalized["question_type"] = "step_back"
            elif normalized.get("num_question"):
                normalized["question_type"] = "other"
        
        return normalized
    
    def _get_empty_template(self) -> Dict[str, Any]:
        """Создание пустого шаблона ответа"""
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