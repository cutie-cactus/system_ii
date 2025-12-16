"""
Рекомендательная система книг
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import Counter
from config import Config

class BookRecommender:
    def __init__(self, metrics_full, metrics_filtered=None):
        """
        Инициализация рекомендательной системы
        
        Args:
            metrics_full: метрики для полного датасета
            metrics_filtered: метрики для отфильтрованного датасета
        """
        self.metrics_full = metrics_full
        self.metrics_filtered = metrics_filtered or metrics_full
        
    def recommend_based_on_likes(self, liked_indices: List[int], disliked_indices: List[int] = None,
                                 n_recommendations: int = Config.DEFAULT_N_RECOMMENDATIONS,
                                 strategy: str = Config.DEFAULT_STRATEGY,
                                 penalty_factor: float = Config.DEFAULT_PENALTY_FACTOR,
                                 weights: Dict[str, float] = None) -> List[Tuple[int, float]]:
        """
        Рекомендация книг на основе лайков/дизлайков
        
        Returns:
            Список кортежей (индекс_книги, оценка_схожести)
        """
        if not liked_indices:
            return []
        
        if disliked_indices is None:
            disliked_indices = []
        
        if weights is None:
            weights = Config.DEFAULT_WEIGHTS
        
        # Выбираем стратегию
        if strategy == 'combined':
            all_scores = self._combined_strategy(liked_indices, weights)
        elif strategy == 'average':
            all_scores = self._average_strategy(liked_indices, weights)
        elif strategy == 'union':
            all_scores = self._union_strategy(liked_indices, weights)
        elif strategy == 'content_boost':
            all_scores = self._content_boost_strategy(liked_indices, weights)
        else:
            all_scores = self._combined_strategy(liked_indices, weights)
        
        # Применяем штраф за дизлайки
        if disliked_indices:
            all_scores = self._apply_dislike_penalty(all_scores, disliked_indices, penalty_factor)
        
        # Исключаем лайки из рекомендаций
        for idx in liked_indices:
            if idx in all_scores:
                del all_scores[idx]
        
        # Сортируем и выбираем лучшие
        recommendations = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def recommend_for_book(self, book_idx: int, n_recommendations: int = 5,
                          weights: Dict[str, float] = None) -> List[Tuple[int, float]]:
        """
        Рекомендации на основе одной книги
        """
        return self.metrics_filtered.get_similar_books(book_idx, n_recommendations, weights)
    
    def compare_books(self, book1_idx: int, book2_idx: int,
                     weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Сравнение двух книг
        """
        if weights is None:
            weights = Config.DEFAULT_WEIGHTS
        
        similarity = self.metrics_full.similarity_score(book1_idx, book2_idx, weights)
        
        book1 = self.metrics_full.df.iloc[book1_idx]
        book2 = self.metrics_full.df.iloc[book2_idx]
        
        # Анализ различий
        differences = []
        if book1['genre'] != book2['genre']:
            genre_dist = self.metrics_full.taxonomic_distance(book1['genre'], book2['genre'])
            differences.append(f"Жанр: {book1['genre']} vs {book2['genre']} (расстояние: {genre_dist:.2f})")
        
        if book1['author'] != book2['author']:
            differences.append(f"Автор: {book1['author']} vs {book2['author']}")
        
        if abs(book1['year'] - book2['year']) > 10:
            differences.append(f"Год издания: {book1['year']} vs {book2['year']}")
        
        if abs(book1['pages'] - book2['pages']) > 100:
            differences.append(f"Объем: {book1['pages']} vs {book2['pages']} страниц")
        
        # Общие черты
        common_features = []
        if book1['genre'] == book2['genre']:
            common_features.append(f"Один жанр: {book1['genre']}")
        if book1['author'] == book2['author']:
            common_features.append(f"Один автор: {book1['author']}")
        if book1['publisher'] == book2['publisher']:
            common_features.append(f"Один издатель: {book1['publisher']}")
        if book1['language'] == book2['language']:
            common_features.append(f"Один язык: {book1['language']}")
        if book1['has_illustrations'] == book2['has_illustrations']:
            common_features.append(f"Наличие иллюстраций: {'Есть' if book1['has_illustrations'] == 1 else 'Нет'}")
        
        return {
            'similarity': similarity,
            'differences': differences,
            'common_features': common_features,
            'book1': {
                'title': book1['title'],
                'author': book1['author'],
                'genre': book1['genre'],
                'year': book1['year'],
                'pages': book1['pages']
            },
            'book2': {
                'title': book2['title'],
                'author': book2['author'],
                'genre': book2['genre'],
                'year': book2['year'],
                'pages': book2['pages']
            }
        }
    
    def _combined_strategy(self, liked_indices: List[int], weights: Dict[str, float]) -> Dict[int, float]:
        """Комбинированная стратегия"""
        book_scores = {}
        
        for book_idx in range(len(self.metrics_filtered.df)):
            if book_idx in liked_indices:
                continue
                
            total_similarity = 0
            for liked_idx in liked_indices:
                similarity = self.metrics_full.similarity_score(book_idx, liked_idx, weights)
                total_similarity += similarity
            
            avg_similarity = total_similarity / len(liked_indices)
            book_scores[book_idx] = avg_similarity
        
        # Усиливаем рекомендации с общими признаками
        return self._boost_by_common_features(liked_indices, book_scores)
    
    def _average_strategy(self, liked_indices: List[int], weights: Dict[str, float]) -> Dict[int, float]:
        """Стратегия усреднения"""
        book_scores = {}
        
        for book_idx in range(len(self.metrics_filtered.df)):
            if book_idx in liked_indices:
                continue
                
            total_similarity = 0
            for liked_idx in liked_indices:
                similarity = self.metrics_full.similarity_score(book_idx, liked_idx, weights)
                total_similarity += similarity
            
            avg_similarity = total_similarity / len(liked_indices)
            book_scores[book_idx] = avg_similarity
        
        return book_scores
    
    def _union_strategy(self, liked_indices: List[int], weights: Dict[str, float]) -> Dict[int, float]:
        """Стратегия объединения"""
        max_scores = {}
        
        for book_idx in range(len(self.metrics_filtered.df)):
            if book_idx in liked_indices:
                continue
                
            max_similarity = 0
            for liked_idx in liked_indices:
                similarity = self.metrics_full.similarity_score(book_idx, liked_idx, weights)
                max_similarity = max(max_similarity, similarity)
            
            max_scores[book_idx] = max_similarity
        
        return max_scores
    
    def _content_boost_strategy(self, liked_indices: List[int], weights: Dict[str, float]) -> Dict[int, float]:
        """Стратегия усиления контента"""
        base_scores = self._average_strategy(liked_indices, weights)
        return self._boost_by_common_features(liked_indices, base_scores)
    
    def _apply_dislike_penalty(self, all_scores: Dict[int, float], disliked_indices: List[int],
                              penalty_factor: float) -> Dict[int, float]:
        """Применение штрафа за дизлайки"""
        penalized_scores = {}
        
        for book_idx, similarity in all_scores.items():
            if book_idx in disliked_indices:
                continue
                
            # Вычисляем схожесть с дизлайками
            max_dislike_similarity = 0
            for dislike_idx in disliked_indices:
                dislike_sim = self.metrics_full.similarity_score(book_idx, dislike_idx)
                max_dislike_similarity = max(max_dislike_similarity, dislike_sim)
            
            # Применяем штраф
            penalty = max_dislike_similarity * penalty_factor
            penalized_similarity = similarity * (1 - penalty)
            
            penalized_scores[book_idx] = max(penalized_similarity, 0)
        
        return penalized_scores
    
    def _boost_by_common_features(self, liked_indices: List[int], book_scores: Dict[int, float]) -> Dict[int, float]:
        """Усиление оценок на основе общих признаков"""
        boosted_scores = book_scores.copy()
        
        # Анализируем общие черты понравившихся книг
        liked_books = [self.metrics_full.df.iloc[idx] for idx in liked_indices]
        
        genres = [book['genre'] for book in liked_books]
        authors = [book['author'] for book in liked_books]
        
        genre_counter = Counter(genres)
        author_counter = Counter(authors)
        
        most_common_genre = genre_counter.most_common(1)[0][0] if genre_counter else None
        most_common_author = author_counter.most_common(1)[0][0] if author_counter else None
        
        # Усиливаем книги с общими признаками
        for book_idx in boosted_scores:
            book = self.metrics_filtered.df.iloc[book_idx]
            
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
    
    def format_recommendations(self, recommendations: List[Tuple[int, float]],
                              liked_indices: List[int] = None,
                              disliked_indices: List[int] = None) -> str:
        """
        Форматирование рекомендаций в читаемый текст
        """
        if not recommendations:
            return "Не найдено подходящих рекомендаций"
        
        output = []
        output.append("=" * 70)
        output.append("РЕКОМЕНДАЦИИ:")
        output.append("=" * 70)
        
        for i, (book_idx, similarity) in enumerate(recommendations, 1):
            book = self.metrics_filtered.df.iloc[book_idx]
            
            output.append(f"{i}. {book['title']} - {book['author']}")
            output.append(f"   Жанр: {book['genre']}, Год: {book['year']}, Страниц: {book['pages']}")
            output.append(f"   Схожесть: {similarity:.3f}")
            
            # Находим наиболее похожую книгу из лайков
            if liked_indices:
                best_match = None
                best_similarity = 0
                for liked_idx in liked_indices:
                    sim = self.metrics_full.similarity_score(book_idx, liked_idx)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_match = self.metrics_full.df.iloc[liked_idx]
                
                if best_match is not None and best_similarity > 0.3:
                    output.append(f"   Похожа на: '{best_match['title']}' (схожесть: {best_similarity:.3f})")
            
            # Проверяем общие черты
            if liked_indices:
                common_features = []
                for liked_idx in liked_indices:
                    liked_book = self.metrics_full.df.iloc[liked_idx]
                    if book['genre'] == liked_book['genre']:
                        common_features.append(f"жанр '{liked_book['genre']}'")
                    if book['author'] == liked_book['author']:
                        common_features.append(f"автор {liked_book['author']}")
                
                if common_features:
                    output.append(f"   Общие черты: {', '.join(set(common_features))}")
            
            output.append("")
        
        return "\n".join(output)