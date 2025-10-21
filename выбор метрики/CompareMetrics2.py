import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Используем Agg бэкенд для избежания проблем с PyCharm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
df = pd.read_csv('DataBooks.csv')

class ThreeMetricsComparator:
    def __init__(self, df):
        self.df = df
        self._setup_scalers()
        self.numerical_features = ['year', 'pages']

    def _setup_scalers(self):
        """Инициализация нормализаторов для числовых признаков"""
        self.scalers = {}
        numerical_features = ['year', 'pages']

        for feature in numerical_features:
            values = self.df[feature].values.reshape(-1, 1)
            self.scalers[feature] = MinMaxScaler().fit(values)

    def get_numerical_vector(self, book_idx):
        """Получает вектор числовых признаков для книги"""
        book = self.df.iloc[book_idx]
        vector = []

        for feature in self.numerical_features:
            value = self.scalers[feature].transform([[book[feature]]])[0][0]
            vector.append(value)

        return np.array(vector)

    # Метрика 1: Евклидово расстояние
    def euclidean_distance(self, i, j):
        vec1 = self.get_numerical_vector(i)
        vec2 = self.get_numerical_vector(j)
        return np.linalg.norm(vec1 - vec2)

    # Метрика 2: Манхэттенское расстояние
    def manhattan_distance(self, i, j):
        vec1 = self.get_numerical_vector(i)
        vec2 = self.get_numerical_vector(j)
        return np.sum(np.abs(vec1 - vec2))

    # Метрика 3: Косинусное расстояние
    def cosine_distance(self, i, j):
        vec1 = self.get_numerical_vector(i)
        vec2 = self.get_numerical_vector(j)
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return 1 - cosine_sim

    def create_distance_matrix(self, distance_function):
        """Создает матрицу расстояний для заданной функции"""
        n = len(self.df)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    distance_matrix[i, j] = 0
                else:
                    dist = distance_function(i, j)
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist

        return distance_matrix

    def compare_all_metrics(self):
        """Сравнивает все три метрики в одном представлении"""
        print("Сравнение трех метрик расстояния для числовых признаков")
        print("=" * 60)

        # Создаем матрицы для всех метрик
        matrices = {
            'Евклидово': self.create_distance_matrix(self.euclidean_distance),
            'Манхэттен': self.create_distance_matrix(self.manhattan_distance),
            'Косинусное': self.create_distance_matrix(self.cosine_distance)
        }

        # Создаем единую фигуру для сравнения
        fig = plt.figure(figsize=(20, 12))

        # 1. Тепловые карты
        for idx, (name, matrix) in enumerate(matrices.items()):
            plt.subplot(2, 3, idx + 1)
            sns.heatmap(matrix, cmap='viridis', square=True,
                        cbar_kws={'label': 'Расстояние', 'shrink': 0.8})
            plt.title(f'{name} расстояние\nТепловая карта', fontweight='bold', fontsize=12)
            plt.xlabel('Индекс книги')
            plt.ylabel('Индекс книги')

        # 2. Распределения расстояний (гистограммы)
        plt.subplot(2, 3, 4)
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

        for idx, (name, matrix) in enumerate(matrices.items()):
            mask = ~np.eye(matrix.shape[0], dtype=bool)
            values = matrix[mask]
            plt.hist(values, bins=25, alpha=0.7, label=name,
                     color=colors[idx], density=True)

        plt.title('Сравнение распределений\nвсех метрик', fontweight='bold', fontsize=12)
        plt.xlabel('Расстояние')
        plt.ylabel('Плотность вероятности')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. Box plot сравнение
        plt.subplot(2, 3, 5)
        box_data = []
        labels = []

        for name, matrix in matrices.items():
            mask = ~np.eye(matrix.shape[0], dtype=bool)
            values = matrix[mask]
            box_data.append(values)
            labels.append(name)

        box_plot = plt.boxplot(box_data, labels=labels, patch_artist=True)

        # Раскрашиваем box plot
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        plt.title('Box plot распределений', fontweight='bold', fontsize=12)
        plt.ylabel('Расстояние')
        plt.grid(True, alpha=0.3)

        # 4. Статистика метрик
        plt.subplot(2, 3, 6)
        plt.axis('off')

        # Собираем статистику
        stats_data = []
        for name, matrix in matrices.items():
            mask = ~np.eye(matrix.shape[0], dtype=bool)
            values = matrix[mask]

            stats = {
                'Метрика': name,
                'Среднее': f'{np.mean(values):.3f}',
                'Стд': f'{np.std(values):.3f}',
                'Медиана': f'{np.median(values):.3f}',
                'Min': f'{np.min(values):.3f}',
                'Max': f'{np.max(values):.3f}'
            }
            stats_data.append(stats)

        # Создаем таблицу
        table_data = [[stats[key] for key in ['Метрика', 'Среднее', 'Стд', 'Медиана', 'Min', 'Max']]
                      for stats in stats_data]

        table = plt.table(cellText=table_data,
                          colLabels=['Метрика', 'Среднее', 'Стд', 'Медиана', 'Min', 'Max'],
                          cellLoc='center',
                          loc='center',
                          bbox=[0.1, 0.1, 0.9, 0.8])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        plt.title('Статистика метрик', fontweight='bold', fontsize=12, y=0.95)

        plt.tight_layout()
        plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("Основной график сохранен как 'metrics_comparison.png'")

        # Создаем второй график для дополнительного анализа
        fig2 = plt.figure(figsize=(15, 10))

        # 1. Попарное сравнение метрик
        comparisons = [
            ('Евклидово', 'Манхэттен', 1),
            ('Евклидово', 'Косинусное', 2),
            ('Манхэттен', 'Косинусное', 3)
        ]

        for (name1, name2, pos) in comparisons:
            plt.subplot(2, 3, pos)

            matrix1 = matrices[name1]
            matrix2 = matrices[name2]

            mask = ~np.eye(matrix1.shape[0], dtype=bool)
            values1 = matrix1[mask]
            values2 = matrix2[mask]

            plt.scatter(values1, values2, alpha=0.6, s=20, color='blue')

            # Линия y=x для сравнения
            max_val = max(np.max(values1), np.max(values2))
            plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, linewidth=1)

            correlation = np.corrcoef(values1, values2)[0, 1]
            plt.title(f'{name1} vs {name2}\ncorr: {correlation:.3f}', fontweight='bold', fontsize=10)
            plt.xlabel(name1)
            plt.ylabel(name2)
            plt.grid(True, alpha=0.3)

        # 2. Анализ вклада признаков
        plt.subplot(2, 3, 4)

        # Анализируем вклад каждого признака в евклидово расстояние
        feature_contributions = []

        for i in range(len(self.df)):
            for j in range(i+1, len(self.df)):
                vec1 = self.get_numerical_vector(i)
                vec2 = self.get_numerical_vector(j)

                # Вклад каждого признака в евклидово расстояние
                contributions = (vec1 - vec2) ** 2
                feature_contributions.append(contributions)

        feature_contributions = np.array(feature_contributions)
        mean_contributions = np.mean(feature_contributions, axis=0)
        total = np.sum(mean_contributions)
        percentages = (mean_contributions / total) * 100

        features = ['Год', 'Страницы', 'Иллюстрации']
        bars = plt.bar(features, percentages, color=['#ff9999', '#66b3ff', '#99ff99'])
        plt.title('Вклад признаков в\nевклидово расстояние', fontweight='bold', fontsize=12)
        plt.ylabel('Процент вклада (%)')

        # Добавляем значения на столбцы
        for bar, percentage in zip(bars, percentages):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.grid(True, alpha=0.3, axis='y')

        # 3. Рейтинг метрик по критериям
        plt.subplot(2, 3, 5)
        plt.axis('off')

        # Оцениваем метрики по критериям
        criteria_weights = {
            'Дифференциация': 0.3,
            'Стабильность': 0.25,
            'Интерпретируемость': 0.25,
            'Устойчивость': 0.2
        }

        metrics_scores = {}
        for name, matrix in matrices.items():
            mask = ~np.eye(matrix.shape[0], dtype=bool)
            values = matrix[mask]

            # Оценки по критериям (0-10)
            differentiation = min(np.std(values) * 10, 10)  # стандартное отклонение
            stability = 10 - min((np.std(values) / np.mean(values)) * 2, 10)  # коэффициент вариации
            interpretability = 10 if name in ['Евклидово', 'Манхэттен'] else 7  # простота интерпретации
            robustness = 8 if name == 'Манхэттен' else (7 if name == 'Евклидово' else 6)  # устойчивость к выбросам

            total_score = (differentiation * criteria_weights['Дифференциация'] +
                           stability * criteria_weights['Стабильность'] +
                           interpretability * criteria_weights['Интерпретируемость'] +
                           robustness * criteria_weights['Устойчивость'])

            metrics_scores[name] = {
                'Дифференциация': differentiation,
                'Стабильность': stability,
                'Интерпретируемость': interpretability,
                'Устойчивость': robustness,
                'Общий балл': total_score
            }

        # Создаем таблицу с оценками
        score_data = []
        for name, scores in metrics_scores.items():
            row = [name] + [f'{scores[key]:.1f}' for key in ['Дифференциация', 'Стабильность', 'Интерпретируемость', 'Устойчивость', 'Общий балл']]
            score_data.append(row)

        score_table = plt.table(cellText=score_data,
                                colLabels=['Метрика', 'Дифф.', 'Стаб.', 'Интер.', 'Устой.', 'Общий'],
                                cellLoc='center',
                                loc='center',
                                bbox=[0, 0.2, 1, 0.6])

        score_table.auto_set_font_size(False)
        score_table.set_fontsize(9)
        score_table.scale(1, 1.5)
        plt.title('Оценка метрик по критериям', fontweight='bold', fontsize=12, y=0.9)

        # 4. Рекомендация лучшей метрики
        plt.subplot(2, 3, 6)
        plt.axis('off')

        # Определяем лучшую метрику
        best_metric = max(metrics_scores.items(), key=lambda x: x[1]['Общий балл'])

        recommendation_text = f"РЕКОМЕНДАЦИЯ:\n\n" \
                              f"Лучшая метрика: {best_metric[0]}\n" \
                              f"Общий балл: {best_metric[1]['Общий балл']:.1f}/10\n\n" \
                              f"Преимущества:\n" \
                              f"- Хорошая дифференциация\n" \
                              f"- Стабильные результаты\n" \
                              f"- Простая интерпретация"

        plt.text(0.1, 0.7, recommendation_text, fontsize=11,
                 verticalalignment='top', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))

        plt.title('Финальная рекомендация', fontweight='bold', fontsize=12, y=0.9)

        plt.tight_layout()
        plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
        print("Детальный анализ сохранен как 'detailed_analysis.png'")

        # Вывод итоговой статистики
        print("\nИТОГОВАЯ СТАТИСТИКА:")
        print("-" * 40)
        for name, scores in metrics_scores.items():
            print(f"{name}: {scores['Общий балл']:.1f}/10")

        print(f"\nЛУЧШАЯ МЕТРИКА: {best_metric[0]} ({best_metric[1]['Общий балл']:.1f}/10)")

        return best_metric[0], matrices

# Запуск сравнения
if __name__ == "__main__":
    comparator = ThreeMetricsComparator(df)
    best_metric, all_matrices = comparator.compare_all_metrics()

    # Дополнительный анализ для лучшей метрики
    print(f"\nДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ ДЛЯ {best_metric}:")
    print("=" * 50)

    best_matrix = all_matrices[best_metric]

    # Находим наиболее похожие и наименее похожие пары книг
    n = len(df)
    similarities = []

    for i in range(n):
        for j in range(i+1, n):
            similarities.append((i, j, best_matrix[i, j]))

    # Сортируем по расстоянию
    similarities.sort(key=lambda x: x[2])

    print("\nСамые похожие книги:")
    for i, (idx1, idx2, dist) in enumerate(similarities[:3]):
        book1 = df.iloc[idx1]
        book2 = df.iloc[idx2]
        print(f"{i+1}. '{book1['title']}' и '{book2['title']}'")
        print(f"   Расстояние: {dist:.3f}")
        print(f"   Общие черты: {book1['genre']}, {book1['year']}г, {book1['pages']}стр")
        print()

    print("\nСамые разные книги:")
    for i, (idx1, idx2, dist) in enumerate(similarities[-3:]):
        book1 = df.iloc[idx1]
        book2 = df.iloc[idx2]
        print(f"{i+1}. '{book1['title']}' и '{book2['title']}'")
        print(f"   Расстояние: {dist:.3f}")
        print(f"   Различия: {book1['genre']} vs {book2['genre']}, "
              f"{book1['year']}г vs {book2['year']}г, "
              f"{book1['pages']}стр vs {book2['pages']}стр")
        print()

    print("\nГрафики сохранены в файлы:")
    print("- metrics_comparison.png: основные графики сравнения")
    print("- detailed_analysis.png: детальный анализ и рекомендации")