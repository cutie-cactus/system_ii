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
                                     exclude_liked=True, strategy='combined', disliked_book_indices=None,
                                     penalty_factor=0.7):
    """
    Рекомендация книг на основе нескольких понравившихся книг
    
    Параметры:
    - metrics: объект BookDistanceMetrics
    - liked_book_indices: список индексов понравившихся книг
    - n_recommendations: количество рекомендаций
    - exclude_liked: исключать ли понравившиеся книги из результатов
    - strategy: стратегия объединения рекомендаций
    - disliked_book_indices: список индексов непонравившихся книг
    - penalty_factor: коэффициент штрафа за схожесть с дизлайками (0-1)
    """
    
    if not liked_book_indices:
        print("Ошибка: список понравившихся книг пуст")
        return []
    
    # Инициализируем список дизлайков если None
    if disliked_book_indices is None:
        disliked_book_indices = []
    
    # Получаем информацию о понравившихся и непонравившихся книгах
    liked_books = [metrics.df.iloc[idx] for idx in liked_book_indices]
    disliked_books = [metrics.df.iloc[idx] for idx in disliked_book_indices]
    
    print("=" * 70)
    print("РЕКОМЕНДАЦИИ НА ОСНОВЕ ВАШИХ ПРЕДПОЧТЕНИЙ:")
    print("=" * 70)
    
    print("👍 ПОНРАВИЛИСЬ:")
    for i, idx in enumerate(liked_book_indices):
        book = metrics.df.iloc[idx]
        print(f"  {i+1}. '{book['title']}' - {book['author']} ({book['genre']})")
    
    if disliked_book_indices:
        print("\n👎 НЕ ПОНРАВИЛИСЬ:")
        for i, idx in enumerate(disliked_book_indices):
            book = metrics.df.iloc[idx]
            print(f"  {i+1}. '{book['title']}' - {book['author']} ({book['genre']})")
    print()
    
    # Получаем оценки для ВСЕХ книг с учетом стратегии
    if strategy == 'combined':
        all_scores = _combined_strategy_all_books(metrics, liked_book_indices, weights, exclude_liked)
    elif strategy == 'average':
        all_scores = _average_strategy_all_books(metrics, liked_book_indices, weights, exclude_liked)
    elif strategy == 'union':
        all_scores = _union_strategy_all_books(metrics, liked_book_indices, weights, exclude_liked)
    elif strategy == 'content_boost':
        all_scores = _content_boost_strategy_all_books(metrics, liked_book_indices, weights, exclude_liked)
    else:
        all_scores = _combined_strategy_all_books(metrics, liked_book_indices, weights, exclude_liked)
    
    # Применяем штраф за дизлайки ко ВСЕМ книгам
    if disliked_book_indices:
        all_scores = _apply_dislike_penalty_all_books(metrics, all_scores, disliked_book_indices, penalty_factor)
    
    # Теперь сортируем ВСЕ книги по скорректированной оценке и выбираем лучшие
    all_recommendations = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    recommendations = all_recommendations[:n_recommendations]
    
    # Выводим рекомендации
    _display_recommendations(metrics, recommendations, liked_books, disliked_books)
    
    return recommendations


def _apply_dislike_penalty_all_books(metrics, all_scores, disliked_indices, penalty_factor):
    """Применяет штраф ко ВСЕМ книгам на основе дизлайков"""
    penalized_scores = {}
    
    for book_idx, similarity in all_scores.items():
        if book_idx in disliked_indices:
            continue  # Полностью исключаем дизлайки
            
        # Вычисляем схожесть с дизлайками
        max_dislike_similarity = 0
        for dislike_idx in disliked_indices:
            dislike_sim = metrics.similarity_score(book_idx, dislike_idx)
            max_dislike_similarity = max(max_dislike_similarity, dislike_sim)
        
        # Применяем штраф
        penalty = max_dislike_similarity * penalty_factor
        penalized_similarity = similarity * (1 - penalty)
        
        penalized_scores[book_idx] = max(penalized_similarity, 0)
    
    return penalized_scores


def _combined_strategy_all_books(metrics, liked_indices, weights, exclude_liked):
    """Комбинированная стратегия: усреднение + усиление по общим признакам для ВСЕХ книг"""
    # Шаг 1: Вычисляем среднее расстояние до всех понравившихся книг для ВСЕХ книг
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
    
    # Шаг 2: Усиливаем рекомендации с общими признаками для ВСЕХ книг
    boosted_scores = _boost_by_common_features(metrics, liked_indices, book_scores)
    
    return boosted_scores


def _average_strategy_all_books(metrics, liked_indices, weights, exclude_liked):
    """Стратегия усреднения: простая средняя схожесть для ВСЕХ книг"""
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
    
    return book_scores


def _union_strategy_all_books(metrics, liked_indices, weights, exclude_liked):
    """Стратегия объединения: берем лучшие рекомендации от каждой книги для ВСЕХ книг"""
    # Создаем временный словарь для хранения максимальной схожести
    max_scores = {}
    
    for book_idx in range(len(metrics.df)):
        if exclude_liked and book_idx in liked_indices:
            continue
            
        # Для каждой книги находим максимальную схожесть с любой понравившейся книгой
        max_similarity = 0
        for liked_idx in liked_indices:
            similarity = metrics.similarity_score(book_idx, liked_idx, weights)
            max_similarity = max(max_similarity, similarity)
        
        max_scores[book_idx] = max_similarity
    
    return max_scores


def _content_boost_strategy_all_books(metrics, liked_indices, weights, exclude_liked):
    """Стратегия усиления контента: усиление рекомендаций с общими признаками для ВСЕХ книг"""
    # Сначала получаем базовые оценки для ВСЕХ книг через усреднение
    base_scores = _average_strategy_all_books(metrics, liked_indices, weights, exclude_liked)
    
    # Усиливаем на основе общих признаков для ВСЕХ книг
    boosted_scores = _boost_by_common_features(metrics, liked_indices, base_scores)
    
    return boosted_scores


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
            boost *= (1 + 0.15 * matching_genres)

        # Усиление за множественные совпадения автора
        matching_authors = sum(1 for liked_author in authors if liked_author == book['author'])
        if matching_authors > 1:
            boost *= (1 + 0.2 * matching_authors)
        
        boosted_scores[book_idx] *= boost
    
    return boosted_scores


def _display_recommendations(metrics, recommendations, liked_books, disliked_books=None):
    """Отображение рекомендаций с анализом"""
    if disliked_books is None:
        disliked_books = []
        
    print("ТОП РЕКОМЕНДАЦИЙ:")
    print("-" * 70)
    
    if not recommendations:
        print("Не найдено подходящих рекомендаций с учетом ваших предпочтений")
        return
    
    for i, (book_idx, similarity) in enumerate(recommendations, 1):
        book = metrics.df.iloc[book_idx]
        
        # Находим наиболее похожие книги из понравившихся
        best_matches = []
        for liked in liked_books:
            liked_idx = metrics.df[metrics.df['title'] == liked['title']].index[0]
            sim = metrics.similarity_score(book_idx, liked_idx)
            best_matches.append((liked['title'], sim))
        
        best_matches.sort(key=lambda x: x[1], reverse=True)
        top_match = best_matches[0] if best_matches else ("", 0)
        
        # Проверяем схожесть с дизлайками
        max_dislike_similarity = 0
        if disliked_books:
            for disliked in disliked_books:
                disliked_idx = metrics.df[metrics.df['title'] == disliked['title']].index[0]
                dislike_sim = metrics.similarity_score(book_idx, disliked_idx)
                max_dislike_similarity = max(max_dislike_similarity, dislike_sim)
        
        print(f"{i}. {book['title']} - {book['author']}")
        print(f"   Жанр: {book['genre']}, Год: {book['year']}, Страниц: {book['pages']}")
        print(f"   Общая схожесть: {similarity:.3f}")
        
        if top_match[1] > 0:
            print(f"   Наиболее похожа на: '{top_match[0]}' (схожесть: {top_match[1]:.3f})")
        
        if max_dislike_similarity > 0.6:
            print(f"   ⚠️  Умеренно похожа на непонравившиеся книги (схожесть: {max_dislike_similarity:.3f})")
        
        # Показываем общие черты с понравившимися книгами
        common_features = []
        for liked in liked_books:
            if book['genre'] == liked['genre']:
                common_features.append(f"жанр '{liked['genre']}'")
            if book['author'] == liked['author']:
                common_features.append(f"автор {liked['author']}")
        
        if common_features:
            print(f"   ✅ Общие черты: {', '.join(set(common_features))}")
        
        print()


def interactive_recommendations(metrics):
    """Интерактивный режим рекомендаций"""
    print("ИНТЕРАКТИВНАЯ СИСТЕМА РЕКОМЕНДАЦИЙ КНИГ")
    print("=" * 50)
    
    # Показываем список всех книг
    for i in range(len(metrics.df)):
        book = metrics.df.iloc[i]
        print(f"{i:2d}. {book['title']} - {book['author']} ({book['genre']})")
    
    while True:
        print("\n" + "="*50)
        print("Введите номера понравившихся книг через запятую (например: 1,3,5)")
        print("Или 'q' для выхода")
        
        user_input = input("\nВаш выбор (лайки): ").strip()
        
        if user_input.lower() == 'q':
            break
        
        try:
            # Парсим ввод пользователя для лайков
            liked_indices = [int(idx.strip()) for idx in user_input.split(',')]
            
            # Проверяем валидность индексов лайков
            valid_liked_indices = []
            for idx in liked_indices:
                if idx in valid_liked_indices:
                    print(f"Предупреждение: индекс {idx} указан несколько раз")
                elif 0 <= idx < len(metrics.df):
                    valid_liked_indices.append(idx)
                else:
                    print(f"Предупреждение: индекс {idx} не существует")
            
            if not valid_liked_indices:
                print("Ошибка: не указано ни одного валидного индекса книги")
                continue
            
            # Запрашиваем дизлайки
            print("\nВведите номера НЕпонравившихся книг через запятую (или Enter чтобы пропустить)")
            dislike_input = input("Ваш выбор (дизлайки): ").strip()
            
            valid_disliked_indices = []
            if dislike_input:
                try:
                    disliked_indices = [int(idx.strip()) for idx in dislike_input.split(',')]
                    for idx in disliked_indices:
                        if idx in valid_disliked_indices:
                            print(f"Предупреждение: индекс {idx} указан несколько раз")
                        elif idx in valid_liked_indices:
                            print(f"Предупреждение: индекс {idx} есть в лайках, игнорируем")
                        elif 0 <= idx < len(metrics.df):
                            valid_disliked_indices.append(idx)
                        else:
                            print(f"Предупреждение: индекс {idx} не существует")
                except ValueError:
                    print("Ошибка ввода дизлайков, будет использован пустой список")
            
            # Запрашиваем коэффициент штрафа
            print("\nВведите коэффициент штрафа за дизлайки (0.0-1.0, по умолчанию 0.7)")
            penalty_input = input("Коэффициент: ").strip()
            
            penalty_factor = 0.7
            if penalty_input:
                try:
                    penalty_factor = float(penalty_input)
                    penalty_factor = max(0.0, min(1.0, penalty_factor))  # Ограничиваем диапазон
                except ValueError:
                    print("Ошибка ввода коэффициента, используется значение по умолчанию 0.7")
            
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
                valid_liked_indices, 
                n_recommendations=5,
                strategy=strategy,
                disliked_book_indices=valid_disliked_indices,
                penalty_factor=penalty_factor
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

# Пример 5: Запуск интерактивного режима
print("\n" + "="*70)
print("=== ИНТЕРАКТИВНЫЙ РЕЖИМ ===")
interactive_recommendations(metrics)

