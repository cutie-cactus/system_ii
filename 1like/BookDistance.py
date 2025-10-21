import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class BookDistanceMetrics:
    def __init__(self, df):
        self.df = df
        self._setup_taxonomy_tree()
        self._setup_scalers()
        self.numerical_features = ['year', 'pages']
    
    def _setup_taxonomy_tree(self):
        """Иерархия жанров в виде дерева"""
        self.genre_hierarchy = {
            'художественная': {
                'проза': {
                    'классика': {},
                    'роман': {
                        'любовный_роман': {},
                        'исторический_роман': {}
                    },
                    'фантастика': {
                        'научная_фантастика': {},
                        'антиутопия': {},
                        'фэнтези': {}
                    },
                    'приключения': {},
                    'детектив': {},
                    'сатира': {}
                },
                'поэзия': {},
                'драма': {}
            },
            'нехудожественная': {
                'биография': {},
                'научная': {},
                'популярная_наука': {},
                'история': {},
                'психология': {}
            },
            'детская': {
                'сказка': {},
                'развивающая': {},
                'приключения_детские': {}
            }
        }
        
        # Сопоставление жанров из датасета с узлами дерева
        self.genre_mapping = {
            'классика': 'художественная.проза.классика',
            'антиутопия': 'художественная.проза.фантастика.антиутопия',
            'фэнтези': 'художественная.проза.фантастика.фэнтези',
            'сказка': 'детская.сказка',
            'роман': 'художественная.проза.роман',
            'детектив': 'художественная.проза.детектив',
            'фантастика': 'художественная.проза.фантастика',
            'приключения': 'художественная.проза.приключения',
            'сатира': 'художественная.проза.сатира'
        }
    
    def _setup_scalers(self):
        """Инициализация нормализаторов для числовых признаков"""
        self.scalers = {}
        
        # Год публикации (чем новее, тем лучше)
        years = self.df['year'].values.reshape(-1, 1)
        self.scalers['year'] = MinMaxScaler().fit(years)
        
        # Возрастное ограничение
        age = self.df['age_restriction'].values.reshape(-1, 1)
        self.scalers['age_restriction'] = MinMaxScaler().fit(age)
        
        # Количество страниц
        pages = self.df['pages'].values.reshape(-1, 1)
        self.scalers['pages'] = MinMaxScaler().fit(pages)
    
    def _get_genre_path(self, genre):
        """Получить путь жанра в дереве"""
        return self.genre_mapping.get(genre, '')
    
    def _find_common_ancestor_level(self, path1, path2):
        """Найти уровень общего предка в дереве"""
        if not path1 or not path2:
            return 0
            
        parts1 = path1.split('.')
        parts2 = path2.split('.')
        
        common_level = 0
        for i in range(min(len(parts1), len(parts2))):
            if parts1[i] == parts2[i]:
                common_level += 1
            else:
                break
        return common_level
    
    def taxonomic_distance(self, genre1, genre2, max_depth=4):
        """
        Таксономическое расстояние между жанрами на основе дерева
        Возвращает значение от 0 (одинаковые жанры) до 1 (совсем разные)
        """
        path1 = self._get_genre_path(genre1)
        path2 = self._get_genre_path(genre2)
        
        if not path1 or not path2:
            return 1.0
        
        if path1 == path2:
            return 0.0
        
        common_level = self._find_common_ancestor_level(path1, path2)
        depth1 = len(path1.split('.'))
        depth2 = len(path2.split('.'))
        
        # Формула таксономического расстояния
        distance = (depth1 + depth2 - 2 * common_level) / (2 * max_depth)
        return min(distance, 1.0)
    
    def get_numerical_vector(self, book_idx):
        """Получает вектор числовых признаков для книги"""
        book = self.df.iloc[book_idx]
        vector = []

        for feature in self.numerical_features:
            value = self.scalers[feature].transform([[book[feature]]])[0][0]
            vector.append(value)

        return np.array(vector)
    
    # Метрика 2: Манхэттенское расстояние
    def manhattan_distance(self, i, j):
        vec1 = self.get_numerical_vector(i)
        vec2 = self.get_numerical_vector(j)
        return np.sum(np.abs(vec1 - vec2))
    
    def categorical_distance(self, val1, val2):
        """
        Расстояние для категориальных признаков
        0 - одинаковые значения, 1 - разные
        """
        return 0.0 if val1 == val2 else 1.0

    def bin_distance(self, val1, val2):
        """
        Расстояние для бинарных признаков
        0 - одинаковые значения, 1 - разные
        """
        return 0.0 if val1 == val2 else 1.0
    
    def composite_distance(self, book1_idx, book2_idx, weights=None):
        """
        Композитное расстояние между двумя книгами
        """
        if weights is None:
            weights = {
                'genre': 0.3,
                'year': 0.2,
                'has_illustrations': 0.15,
                'pages': 0.15,
                'author': 0.1,
                'publisher': 0.05,
                'language': 0.05,
                'numerical': 0.35
            }

        book1 = self.df.iloc[book1_idx]
        book2 = self.df.iloc[book2_idx]
        
        total_distance = 0.0
        
        # Таксономическое расстояние по жанру
        genre_dist = self.taxonomic_distance(book1['genre'], book2['genre'])
        total_distance += weights['genre'] * genre_dist
        
        # Числовые признаки
        numerical_dist = self.manhattan_distance(book1_idx, book2_idx)
        total_distance += weights['numerical'] * numerical_dist
        
        # Категориальные признаки
        author_dist = self.categorical_distance(book1['author'], book2['author'])
        total_distance += weights['author'] * author_dist
        
        publisher_dist = self.categorical_distance(book1['publisher'], book2['publisher'])
        total_distance += weights['publisher'] * publisher_dist
        
        language_dist = self.categorical_distance(book1['language'], book2['language'])
        total_distance += weights['language'] * language_dist

        # Бинарные признаки
        ill_dist = self.bin_distance(book1['has_illustrations'], book2['has_illustrations'])
        total_distance += weights['has_illustrations'] * ill_dist

        return total_distance
    
    def similarity_score(self, book1_idx, book2_idx, weights=None):
        """
        Оценка схожести между книгами (1 - distance)
        """
        distance = self.composite_distance(book1_idx, book2_idx, weights)
        return 1.0 - distance
    
    def get_similar_books(self, book_idx, n=5, weights=None):
        """
        Найти n наиболее похожих книг
        """
        similarities = []
        
        for i in range(len(self.df)):
            if i != book_idx:
                sim = self.similarity_score(book_idx, i, weights)
                similarities.append((i, sim))
        
        # Сортировка по убыванию схожести
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]

