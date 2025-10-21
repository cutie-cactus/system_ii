from BookDistance import BookDistanceMetrics
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import Counter


def create_distance_matrix(metrics, weights=None):
    """Создание матрицы расстояний между всеми книгами"""
    n = len(metrics.df)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = metrics.composite_distance(i, j, weights)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix


def recommend_books(metrics, target_book_idx, n_recommendations=5, weights=None):
    """Рекомендация книг на основе схожести с одной книгой"""
    similar_books = metrics.get_similar_books(target_book_idx, n_recommendations, weights)

    target_book = metrics.df.iloc[target_book_idx]
    print(f"Рекомендации для '{target_book['title']}' ({target_book['author']}):")
    print(f"  Жанр: {target_book['genre']}")

    print("-" * 50)

    for idx, similarity in similar_books:
        book = metrics.df.iloc[idx]
        print(f"• {book['title']} ({book['author']})")
        print(f"  Жанр: {book['genre']}, Схожесть: {similarity:.3f}")
        print()


def recommend_based_on_multiple_likes(metrics, liked_book_indices, n_recommendations=10, weights=None, 
                                     exclude_liked=True, strategy='combined'):
    """
    Рекомендация книг на основе нескольких понравившихся книг
    
    Параметры:
    - metrics: объект BookDistanceMetrics
    - liked_book_indices: список индексов понравившихся книг
    - n_recommendations: количество рекомендаций
    - exclude_liked: исключать ли понравившиеся книги из результатов
    - strategy: стратегия объединения рекомендаций
        'combined' - комбинированный подход (по умолчанию)
        'average' - усреднение расстояний
        'union' - объединение рекомендаций от каждой книги
        'content_boost' - усиление рекомендаций на основе общего контента
    """
    
    if not liked_book_indices:
        print("Ошибка: список понравившихся книг пуст")
        return []
    
    # Получаем информацию о понравившихся книгах
    liked_books = [metrics.df.iloc[idx] for idx in liked_book_indices]
    
    print("=" * 70)
    print("РЕКОМЕНДАЦИИ НА ОСНОВЕ ВАШИХ ПОНРАВИВШИХСЯ КНИГ:")
    print("=" * 70)
    
    for i, idx in enumerate(liked_book_indices):
        book = metrics.df.iloc[idx]
        print(f"{i+1}. '{book['title']}' - {book['author']} ({book['genre']})")
    print()
    
    if strategy == 'combined':
        recommendations = _combined_strategy(metrics, liked_book_indices, n_recommendations, weights, exclude_liked)
    elif strategy == 'average':
        recommendations = _average_strategy(metrics, liked_book_indices, n_recommendations, weights, exclude_liked)
    elif strategy == 'union':
        recommendations = _union_strategy(metrics, liked_book_indices, n_recommendations, weights, exclude_liked)
    elif strategy == 'content_boost':
        recommendations = _content_boost_strategy(metrics, liked_book_indices, n_recommendations, weights, exclude_liked)
    else:
        recommendations = _combined_strategy(metrics, liked_book_indices, n_recommendations, weights, exclude_liked)
    
    # Выводим рекомендации
    _display_recommendations(metrics, recommendations, liked_books)
    
    return recommendations


def _combined_strategy(metrics, liked_indices, n_recommendations, weights, exclude_liked):
    """Комбинированная стратегия: усреднение + усиление по общим признакам"""
    # Шаг 1: Вычисляем среднее расстояние до всех понравившихся книг
    book_scores = {}
    
    for book_idx in range(len(metrics.df)):
        if exclude_liked and book_idx in liked_indices:
            continue
            
        total_similarity = 0
        for liked_idx in liked_indices:
            similarity = metrics.similarity_score(book_idx, liked_idx, weights)
            total_similarity += similarity
        
        avg_similarity = total_similarity / len(liked_indices)
        book_scores[book_idx] = avg_similarity
    
    # Шаг 2: Усиливаем рекомендации с общими признаками
    boosted_scores = _boost_by_common_features(metrics, liked_indices, book_scores)
    
    # Сортируем по убыванию схожести
    recommendations = sorted(boosted_scores.items(), key=lambda x: x[1], reverse=True)
    
    return recommendations[:n_recommendations]


def _average_strategy(metrics, liked_indices, n_recommendations, weights, exclude_liked):
    """Стратегия усреднения: простая средняя схожесть"""
    book_scores = {}
    
    for book_idx in range(len(metrics.df)):
        if exclude_liked and book_idx in liked_indices:
            continue
            
        total_similarity = 0
        for liked_idx in liked_indices:
            similarity = metrics.similarity_score(book_idx, liked_idx, weights)
            total_similarity += similarity
        
        avg_similarity = total_similarity / len(liked_indices)
        book_scores[book_idx] = avg_similarity
    
    # Сортируем по убыванию схожести
    recommendations = sorted(book_scores.items(), key=lambda x: x[1], reverse=True)
    
    return recommendations[:n_recommendations]


def _union_strategy(metrics, liked_indices, n_recommendations, weights, exclude_liked):
    """Стратегия объединения: берем лучшие рекомендации от каждой книги"""
    all_recommendations = []
    
    for liked_idx in liked_indices:
        # Получаем рекомендации для каждой понравившейся книги
        similar = metrics.get_similar_books(liked_idx, n_recommendations * 2, weights)
        all_recommendations.extend(similar)
    
    # Объединяем и усредняем схожести для одинаковых книг
    merged_scores = {}
    for book_idx, similarity in all_recommendations:
        if exclude_liked and book_idx in liked_indices:
            continue
        if book_idx in merged_scores:
            merged_scores[book_idx] = max(merged_scores[book_idx], similarity)
        else:
            merged_scores[book_idx] = similarity
    
    # Сортируем по убыванию схожести
    recommendations = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
    
    return recommendations[:n_recommendations]


def _content_boost_strategy(metrics, liked_indices, n_recommendations, weights, exclude_liked):
    """Стратегия усиления контента: усиление рекомендаций с общими признаками"""
    # Сначала получаем базовые рекомендации через усреднение
    base_recommendations = _average_strategy(metrics, liked_indices, n_recommendations * 3, weights, exclude_liked)
    
    # Усиливаем на основе общих признаков
    boosted_scores = _boost_by_common_features(metrics, liked_indices, dict(base_recommendations))
    
    # Сортируем по убыванию схожести
    recommendations = sorted(boosted_scores.items(), key=lambda x: x[1], reverse=True)
    
    return recommendations[:n_recommendations]


def _boost_by_common_features(metrics, liked_indices, book_scores):
    """Усиление оценок на основе общих признаков с понравившимися книгами"""
    boosted_scores = book_scores.copy()
    
    # Анализируем общие черты понравившихся книг
    liked_books = [metrics.df.iloc[idx] for idx in liked_indices]
    
    # Находим наиболее частые признаки
    genres = [book['genre'] for book in liked_books]
    authors = [book['author'] for book in liked_books]
    
    genre_counter = Counter(genres)
    author_counter = Counter(authors)
    
    most_common_genre = genre_counter.most_common(1)[0][0] if genre_counter else None
    most_common_author = author_counter.most_common(1)[0][0] if author_counter else None
    
    # Усиливаем книги с общими признаками
    for book_idx in boosted_scores:
        book = metrics.df.iloc[book_idx]
        
        boost = 1.0
        
        # Усиление за общий жанр
        if most_common_genre and book['genre'] == most_common_genre:
            boost *= 1.2
        
        # Усиление за общего автора
        if most_common_author and book['author'] == most_common_author:
            boost *= 1.3
        
        # Усиление за множественные совпадения жанров
        matching_genres = sum(1 for liked_genre in genres if liked_genre == book['genre'])
        if matching_genres > 1:
            boost *= (1 + 0.1 * matching_genres)
        
        boosted_scores[book_idx] *= boost
    
    return boosted_scores


def _display_recommendations(metrics, recommendations, liked_books):
    """Отображение рекомендаций с анализом"""
    print("ТОП РЕКОМЕНДАЦИЙ:")
    print("-" * 70)
    
    for i, (book_idx, similarity) in enumerate(recommendations, 1):
        book = metrics.df.iloc[book_idx]
        
        # Находим наиболее похожие книги из понравившихся
        best_matches = []
        for liked in liked_books:
            liked_idx = metrics.df[metrics.df['title'] == liked['title']].index[0]
            sim = metrics.similarity_score(book_idx, liked_idx)
            best_matches.append((liked['title'], sim))
        
        best_matches.sort(key=lambda x: x[1], reverse=True)
        top_match = best_matches[0]
        
        print(f"{i}. {book['title']} - {book['author']}")
        print(f"   Жанр: {book['genre']}, Год: {book['year']}, Страниц: {book['pages']}")
        print(f"   Общая схожесть: {similarity:.3f}")
        print(f"   Наиболее похожа на: '{top_match[0]}' (схожесть: {top_match[1]:.3f})")
        
        # Показываем общие черты с понравившимися книгами
        common_features = []
        for liked in liked_books:
            if book['genre'] == liked['genre']:
                common_features.append(f"жанр '{liked['genre']}'")
            if book['author'] == liked['author']:
                common_features.append(f"автор {liked['author']}")
        
        if common_features:
            print(f"   Общие черты: {', '.join(set(common_features))}")
        
        print()


def interactive_recommendations(metrics):
    """Интерактивный режим рекомендаций"""
    print("ИНТЕРАКТИВНАЯ СИСТЕМА РЕКОМЕНДАЦИЙ КНИГ")
    print("=" * 50)
    
    # Показываем список всех книг
    for i in range(len(metrics.df)):
        book = metrics.df.iloc[i]
        print(f"{i:2d}. {book['title']} - {book['author']} ({book['genre']})")
    
    print("\nВведите номера понравившихся книг через запятую (например: 1,3,5)")
    print("Или 'q' для выхода")
    
    while True:
        user_input = input("\nВаш выбор: ").strip()
        
        if user_input.lower() == 'q':
            break
        
        try:
            # Парсим ввод пользователя
            liked_indices = [int(idx.strip()) for idx in user_input.split(',')]
            
            # Проверяем валидность индексов
            valid_indices = []
            for idx in liked_indices:
                if idx in valid_indices:
                    print(f"Предупреждение: индекс {idx} указан несколько раз")
                elif 0 <= idx < len(metrics.df):
                    valid_indices.append(idx)
                else:
                    print(f"Предупреждение: индекс {idx} не существует")
            
            if not valid_indices:
                print("Ошибка: не указано ни одного валидного индекса книги")
                continue
            
            print("\n" + "="*70)
            print("Выберите стратегию рекомендаций:")
            print("1. Комбинированная (рекомендуется)")
            print("2. Усреднение")
            print("3. Объединение") 
            print("4. Контент-усиление")
            
            strategy_choice = input("Ваш выбор (1-4, по умолчанию 1): ").strip()
            
            strategies = {
                '1': 'combined',
                '2': 'average', 
                '3': 'union',
                '4': 'content_boost'
            }
            
            strategy = strategies.get(strategy_choice, 'combined')
            
            # Получаем рекомендации
            recommend_based_on_multiple_likes(
                metrics, 
                valid_indices, 
                n_recommendations=3,
                strategy=strategy
            )
            
            print("Хотите попробовать другие книги? (y/n)")
            continue_choice = input().strip().lower()
            if continue_choice != 'y':
                break
                
        except ValueError:
            print("Ошибка: введите числа через запятую")
        except Exception as e:
            print(f"Произошла ошибка: {e}")


# Загрузка данных
df = pd.read_csv('../DataBooks.csv')

# Пример использования рекомендаций
metrics = BookDistanceMetrics(df)

# Пример 1: Рекомендации для одной книги
print("=== РЕКОМЕНДАЦИИ ДЛЯ ОДНОЙ КНИГИ ===")
recommend_books(metrics, 2, 5)

# Пример 2: Рекомендации на основе нескольких лайков
print("\n" + "="*70)
print("=== РЕКОМЕНДАЦИИ НА ОСНОВЕ НЕСКОЛЬКИХ КНИГ ===")
liked_books = [0, 2, 5]  # Индексы понравившихся книг
recommend_based_on_multiple_likes(metrics, liked_books, n_recommendations=6)

# Пример 3: Сравнение разных стратегий
print("\n" + "="*70)
print("=== СРАВНЕНИЕ СТРАТЕГИЙ ===")
test_likes = [1, 3, 7]

strategies = ['combined', 'average', 'union', 'content_boost']
for strategy in strategies:
    print(f"\n--- Стратегия: {strategy.upper()} ---")
    recommend_based_on_multiple_likes(metrics, test_likes, n_recommendations=3, strategy=strategy)

# Пример 4: Запуск интерактивного режима
print("\n" + "="*70)
print("=== ИНТЕРАКТИВНЫЙ РЕЖИМ ===")
interactive_recommendations(metrics)

