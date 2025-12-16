"""
Конфигурация системы рекомендаций книг
"""

class Config:
    # Пути к файлам
    DATA_PATH = '../DataBooks.csv'
    PROMPT_PATH = 'mainPromtOnStart.md'
    
    # Настройки нейросети
    NEURAL_URL = "http://localhost:11434"
    NEURAL_MODEL = "llama3.1:8b-instruct-q4_0"
    NEURAL_TIMEOUT = 180
    
    # Настройки метрик
    DEFAULT_WEIGHTS = {
        'genre': 0.35,
        'has_illustrations': 0.15,
        'author': 0.2,
        'publisher': 0.05,
        'language': 0.05,
        'numerical': 0.2
    }
    
    # Настройки рекомендаций
    DEFAULT_N_RECOMMENDATIONS = 5
    DEFAULT_STRATEGY = 'combined'
    DEFAULT_PENALTY_FACTOR = 0.7
    
    # Настройки фильтрации
    MAX_RESULTS = 100
    SIMILARITY_THRESHOLD = 0.3