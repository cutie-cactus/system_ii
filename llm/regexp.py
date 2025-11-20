import re
import json
from datetime import datetime

class ImprovedBookQueryParser:
    def __init__(self):
        # Более точные паттерны для определения типа вопроса
        self.patterns = {
            'recommendation': [
                r'(^|\\s)(посоветуй|порекомендуй|подбери|хочу|что.*почитать|что.*предложишь)',
                r'(мне нравятся|люблю|нравится|интересен)',
                r'(не люблю|не нравятся|непохож|не интересен)',
                r'(похож.*на|стиль|в духе|наподобие)',
                r'(порекомендуй|что.*выбрать)'
            ],
            'search': [
                r'(найди|покажи|ищи|выведи|есть ли|найти|поиск)',
                r'(книг[аиу]|произведени[ея]|работы)',
                r'(жанр[е]?|автор[а]?|год[у]?|язык[е]?)',
                r'(последние|новинк[и]?|после|вышл[аио])'
            ],
            'comparison': [
                r'(сравни|отличается|разница|сопоставь|различ)',
                r'(какая.*книг[а]?|что.*лучше|выбрать|предпочти)',
                r'(длиннее|объем|страниц|больше)',
                r'(одного жанра|новее|возраст|рейтинг)',
                r'(чем.*отлич)'
            ],
            'general': [
                r'(сколько.*страниц|объем|длина|число.*страниц)',
                r'(кто.*издатель|издательство|опубликовал)',
                r'(есть ли.*иллюстрац|картинк|рисунк)',
                r'(какой.*жанр|жанровая|относится.*к.*жанру)',
                r'(на каком.*язык|язык.*оригинал|язык.*написания)'
            ],
            'history': [
                r'(вернись|назад|передумал|отмени|отмена)',
                r'(предыдущ|последн|раньше|истори)',
                r'(покажи.*снова|результат.*по|снова.*покажи)'
            ],
            'other': [
                r'^(привет|здравствуй|добрый|hello|hi)',
                r'(что.*умеешь|функц|возможност|умеешь.*делать)',
                r'(спасибо|благодар|помог|отлично)',
                r'(помощь|help|как.*пользовать|инструкц)',
                r'(пока|до свидани|выход|завершить)'
            ]
        }
        
        # Улучшенные паттерны для извлечения параметров
        self.param_patterns = {
            'author': [
                r'(?:автор[ау]?|писатель[я]?)\s*(?:—|:)?\s*["«]?([^"»,.!?]+(?:\s+[^"»,.!?]+)*)["»]?',
                r'((?:не\s+)?нравится|(?:не\s+)?люблю)\s+([^",.!?]+(?:\s+[^"»,.!?]+)*)',
                r'([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)(?:\s|$|,)'
            ],
            'genre': [
                r'жанр[еау]?\s*(?:—|:)?\s*["«]?([^"»,.!?]+)["»]?',
                r'в\s+жанре\s+([^,.!?]+)',
                r'((?:научная\s+)?фантастика|детектив|триллер|классика|фэнтези|роман|драма)'
            ],
            'year': [
                r'(\d{4})\s*года?',
                r'после\s+(\d{4})',
                r'с\s+(\d{4})'
            ],
            'year_range': [
                r'последние\s+(\d+)\s+лет',
                r'за\s+(\d+)\s+лет',
                r'(\d+)\s+лет\s+назад'
            ],
            'pages': [
                r'(\d+)\s*страниц',
                r'страниц[ы]?\s*:\s*(\d+)',
                r'объем.*?(\d+)\s*стр'
            ],
            'language': [
                r'на\s+([а-яё]+)\s+языке',
                r'язык[е]?\s*:\s*([а-яё]+)',
                r'([а-яё]+)\s+язык'
            ],
            'age': [
                r'(\d+)\+',
                r'(\d+)\s*лет',
                r'для\s+(\d+)[\s\-]?летних'
            ],
            'illustrations': [
                r'(иллюстраци[ейя]|картинк[и]|рисунк[и])',
                r'(с\s+иллюстрациями|с\s+картинками)'
            ],
            'book_title': [
                r'["«]([^"»]+)["»]',
                r'книг[аиу]?\s+["«]([^"»]+)["»]',
                r'произведени[ея]?\s+["«]([^"»]+)["»]'
            ],
            'publisher': [
                r'издательств[оа]?\s*(?:—|:)?\s*["«]?([^"»,.!?]+)["»]?',
                r'издал[аи]?\s+["«]?([^"»,.!?]+)["»]?',
                r'опубликовал[аи]?\s+["«]?([^"»,.!?]+)["»]?'
            ]
        }

    def parse_query(self, query):
        query_lower = query.lower().strip()
        
        # Определяем тип вопроса
        question_type = self._detect_question_type(query_lower)
        
        # Извлекаем параметры фильтрации
        filters = self._extract_filters(query_lower, question_type)
        
        # Извлекаем параметры сравнения
        compare = self._extract_comparison(query_lower, question_type)
        
        # Извлекаем feedback
        feedback = self._extract_feedback(query_lower, question_type)
        
        return {
            "question_type": question_type,
            "filter": filters,
            "compare": compare,
            "feedback": feedback
        }

    def _detect_question_type(self, query):
        scores = {q_type: 0 for q_type in self.patterns}
        
        for q_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query)
                if matches:
                    scores[q_type] += len(matches)
        
        # Дополнительная логика для уточнения типа
        if "сколько" in query and "страниц" in query:
            scores['general'] += 2
        
        if "сравни" in query or "отличается" in query:
            scores['comparison'] += 2
            
        if "похож" in query and ("на" in query or "как" in query):
            scores['recommendation'] += 2
            
        if "не люблю" in query or "не нравится" in query:
            scores['recommendation'] += 1
            
        # Выбираем тип с наибольшим счетом
        max_score = max(scores.values())
        if max_score == 0:
            return "search"  # default type
            
        for q_type, score in scores.items():
            if score == max_score:
                return q_type

    def _extract_filters(self, query, question_type):
        filters = {
            "author": "",
            "publisher": "",
            "year_from": "",
            "year_to": "",
            "language": "",
            "age_restriction": "",
            "genre": "",
            "pages_from": "",
            "pages_to": "",
            "has_illustrations": ""
        }
        
        # Поиск автора
        for pattern in self.param_patterns['author']:
            matches = re.findall(pattern, query)
            for match in matches:
                if isinstance(match, tuple):
                    for item in match:
                        if item and len(item) > 3 and not any(word in item for word in ['нравится', 'люблю']):
                            filters['author'] = item.strip()
                            break
                elif match and len(match) > 3:
                    filters['author'] = match.strip()
                    break
            if filters['author']:
                break
        
        # Поиск жанра
        for pattern in self.param_patterns['genre']:
            match = re.search(pattern, query)
            if match:
                filters['genre'] = match.group(1).strip()
                break
        
        # Поиск года
        for pattern in self.param_patterns['year']:
            match = re.search(pattern, query)
            if match:
                year = match.group(1)
                if 'после' in query or 'с' in query:
                    filters['year_from'] = year
                else:
                    filters['year_from'] = year
                    filters['year_to'] = year
                break
        
        # Поиск диапазона лет
        for pattern in self.param_patterns['year_range']:
            match = re.search(pattern, query)
            if match:
                years = int(match.group(1))
                current_year = datetime.now().year
                filters['year_from'] = str(current_year - years)
                filters['year_to'] = str(current_year)
                break
        
        # Поиск языка
        for pattern in self.param_patterns['language']:
            match = re.search(pattern, query)
            if match:
                filters['language'] = match.group(1)
                break
        
        # Поиск возрастного ограничения
        for pattern in self.param_patterns['age']:
            match = re.search(pattern, query)
            if match:
                filters['age_restriction'] = match.group(1)
                break
        
        # Поиск иллюстраций
        for pattern in self.param_patterns['illustrations']:
            if re.search(pattern, query):
                filters['has_illustrations'] = "true"
                break
        
        # Поиск издателя
        for pattern in self.param_patterns['publisher']:
            match = re.search(pattern, query)
            if match:
                filters['publisher'] = match.group(1).strip()
                break
        
        return filters

    def _extract_comparison(self, query, question_type):
        compare = {
            "count_books": "",
            "title1": "",
            "author1": "",
            "title2": "",
            "author2": "",
            "title3": "",
            "author3": ""
        }
        
        # Поиск названий книг для сравнения
        book_titles = []
        for pattern in self.param_patterns['book_title']:
            matches = re.findall(pattern, query)
            book_titles.extend(matches)
        
        if book_titles:
            compare['count_books'] = str(len(book_titles))
            for i, title in enumerate(book_titles[:3], 1):
                compare[f'title{i}'] = title.strip()
        
        return compare

    def _extract_feedback(self, query, question_type):
        feedback = {
            "likes": [],
            "dislikes": []
        }
        
        # Поиск книг, которые нравятся
        like_patterns = [
            r'(нравятся|люблю|нравится|интересен)\s+["«]([^"»]+)["»]',
            r'похож[аеи]?\s+на\s+["«]([^"»]+)["»]',
            r'стил[еья]?\s+["«]([^"»]+)["»]'
        ]
        
        for pattern in like_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if isinstance(match, tuple):
                    book = match[1] if len(match) > 1 else match[0]
                else:
                    book = match
                if book and book not in feedback['likes']:
                    feedback['likes'].append(book.strip())
        
        # Поиск авторов/книг, которые не нравятся
        dislike_patterns = [
            r'(не\s+нравится|не\s+люблю|не\s+интересен)\s+([^",.!?]+(?:\s+[^"»,.!?]+)*)',
            r'(не\s+нравится|не\s+люблю)\s+["«]([^"»]+)["»]',
            r'непохож[аеи]?\s+на\s+["«]([^"»]+)["»]'
        ]
        
        for pattern in dislike_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if isinstance(match, tuple):
                    item = match[1] if len(match) > 1 else match[0]
                else:
                    item = match
                if item and item not in feedback['dislikes']:
                    feedback['dislikes'].append(item.strip())
        
        # Извлекаем книги из кавычек для рекомендаций
        if question_type == 'recommendation':
            for pattern in self.param_patterns['book_title']:
                matches = re.findall(pattern, query)
                for book in matches:
                    if book and book not in feedback['likes'] and book not in feedback['dislikes']:
                        # Если есть отрицание, добавляем в dislikes, иначе в likes
                        if re.search(r'не\s+', query.split(book)[0][-50:] if book in query else ''):
                            feedback['dislikes'].append(book.strip())
                        else:
                            feedback['likes'].append(book.strip())
        
        return feedback

# Пример использования
def process_query(query):
    parser = ImprovedBookQueryParser()
    result = parser.parse_query(query)
    return json.dumps(result, ensure_ascii=False, indent=2)

# Тестовые примеры
if __name__ == "__main__":
    test_queries = [
        'Посоветуй что-нибудь похожее на "Гарри Поттер и философский камень"',
        'Найди книги в жанре научная фантастика',
        'Сравни "Война и мир" и "Анна Каренина"',
        'Сколько страниц в книге "Властелин колец"?',
        'Мне не нравится Лев Толстой',
        'Кто издатель книги "1984"?',
        'Найди книги на французском языке',
        'Покажи книги для подростков 16+ с иллюстрациями'
    ]
    
    for query in test_queries:
        print(f"Запрос: {query}")
        print(process_query(query))
        print("-" * 50)