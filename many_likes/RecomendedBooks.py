from BookDistance import BookDistanceMetrics
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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
    """Рекомендация книг на основе схожести"""
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


# Загрузка данных
df = pd.read_csv('../DataBooks.csv')

# Пример использования рекомендаций
metrics = BookDistanceMetrics(df)
recommend_books(metrics, 2, 5)
