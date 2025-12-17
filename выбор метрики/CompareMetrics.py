import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import MDS
from scipy.spatial.distance import euclidean, cityblock, cosine
import warnings
from BookDistance import BookDistanceMetrics
warnings.filterwarnings('ignore')

# Загрузка данных
df = pd.read_csv('../DataBooks.csv')

class NumericalDistanceComparator:
    def __init__(self, df):
        self.df = df
        self._setup_scalers()
        
        # Используем только числовые признаки: year, pages
        self.numerical_features = ['year', 'pages']
        
    def _setup_scalers(self):
        """Инициализация нормализаторов для числовых признаков"""
        self.scalers = {}
        self.standard_scalers = {}
        
        numerical_features = ['year', 'pages']
        
        for feature in numerical_features:
            values = self.df[feature].values.reshape(-1, 1)
            self.scalers[feature] = MinMaxScaler().fit(values)  # для нормализации [0,1]
            self.standard_scalers[feature] = StandardScaler().fit(values)  # для стандартизации
    
    def get_numerical_vector(self, book_idx, normalized=True):
        """Получает вектор числовых признаков для книги"""
        book = self.df.iloc[book_idx]
        vector = []
        
        for feature in self.numerical_features:
            if normalized:
                value = self.scalers[feature].transform([[book[feature]]])[0][0]
            else:
                value = book[feature]
            vector.append(value)
        
        return np.array(vector)
    
    def get_standardized_vector(self, book_idx):
        """Получает стандартизованный вектор числовых признаков"""
        book = self.df.iloc[book_idx]
        vector = []
        
        for feature in self.numerical_features:
            value = self.standard_scalers[feature].transform([[book[feature]]])[0][0]
            vector.append(value)
        
        return np.array(vector)

    # ВАРИАНТ 1: Евклидово расстояние (нормализованное)
    def distance_v1_euclidean(self, i, j):
        """
        Классическое евклидово расстояние по нормализованным признакам
        """
        vec1 = self.get_numerical_vector(i)
        vec2 = self.get_numerical_vector(j)
        
        return euclidean(vec1, vec2)
    
    # ВАРИАНТ 2: Манхэттенское расстояние
    def distance_v2_manhattan(self, i, j):
        """
        Манхэттенское расстояние (L1 норма)
        Хорошо для данных с выбросами
        """
        vec1 = self.get_numerical_vector(i)
        vec2 = self.get_numerical_vector(j)
        
        return cityblock(vec1, vec2)
    
    # ВАРИАНТ 3: Косинусное расстояние
    def distance_v3_cosine(self, i, j):
        """
        Косинусное расстояние (1 - cosine similarity)
        Учитывает направление, а не величину
        """
        vec1 = self.get_numerical_vector(i)
        vec2 = self.get_numerical_vector(j)
        
        # Косинусное сходство
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        # Преобразуем в расстояние
        return 1 - cosine_sim
    
    # ВАРИАНТ 4: Взвешенное евклидово расстояние
    def distance_v4_weighted_euclidean(self, i, j):
        """
        Евклидово расстояние с весами признаков
        Веса: год - 0.5, страницы - 0.5
        """
        # weights = np.array([0.5, 0.5])  # год, страницы
        
        vec1 = self.get_numerical_vector(i)
        vec2 = self.get_numerical_vector(j)
        
        weighted_diff = (vec1 - vec2)
        return np.sqrt(np.sum(weighted_diff ** 2))
    
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
    
    def evaluate_numerical_metrics(self):
        """Сравнивает все числовые метрики и визуализирует результаты"""
        print("Создание матриц расстояний для числовых признаков...")
        print("Используемые признаки: year, pages, has_illustrations")
        
        # Создаем матрицы для всех метрик
        matrices = {}
        assd = BookDistanceMetrics(df)
        matrices['Евклидово (V1)'] = self.create_distance_matrix(self.distance_v1_euclidean)
        matrices['Манхэттенское (V2)'] = self.create_distance_matrix(self.distance_v2_manhattan)
        matrices['Косинусное (V3)'] = self.create_distance_matrix(self.distance_v3_cosine)
        matrices['Итоговая оценка схожести'] = self.create_distance_matrix(assd.similarity_score)
        
        # Визуализация
        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        fig.suptitle('Сравнение метрик расстояния для ЧИСЛОВЫХ признаков (год, страницы, иллюстрации)', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Тепловые карты
        for idx, (name, matrix) in enumerate(list(matrices.items())[:3]):
            ax = axes[0, idx]
            im = ax.imshow(matrix, cmap='viridis', aspect='auto')
            ax.set_title(f'{name}\nМатрица расстояний', fontweight='bold', fontsize=12)
            ax.set_xlabel('Индекс книги')
            ax.set_ylabel('Индекс книги')
            plt.colorbar(im, ax=ax)
        
        # Статистика по метрикам
        metrics_stats = []
        for name, matrix in matrices.items():
            # Исключаем диагональ для статистики
            mask = ~np.eye(matrix.shape[0], dtype=bool)
            values = matrix[mask]
            
            stats = {
                'Метрика': name,
                'Среднее': np.mean(values),
                'Станд. отклонение': np.std(values),
                'Медиана': np.median(values),
                'Min': np.min(values),
                'Max': np.max(values),
                'Диапазон': np.max(values) - np.min(values),
                'Q1': np.percentile(values, 25),
                'Q3': np.percentile(values, 75)
            }
            metrics_stats.append(stats)
        
        stats_df = pd.DataFrame(metrics_stats)
        
        # Визуализация статистики
        ax = axes[0, 2]
        metrics_names = stats_df['Метрика'].tolist()
        means = stats_df['Среднее'].tolist()
        stds = stats_df['Станд. отклонение'].tolist()
        
        x_pos = np.arange(len(metrics_names))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                     color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'], alpha=0.8)
        ax.set_title('Среднее расстояние и стандартное отклонение', fontweight='bold')
        ax.set_ylabel('Расстояние')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics_names, rotation=45, ha='right')
        
        # Добавляем значения на столбцы
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Распределения расстояний
        ax = axes[1, 0]
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        for idx, (name, matrix) in enumerate(matrices.items()):
            mask = ~np.eye(matrix.shape[0], dtype=bool)
            values = matrix[mask]
            ax.hist(values, alpha=0.7, label=name, bins=25, color=colors[idx])
        
        ax.set_title('Распределения расстояний', fontweight='bold')
        ax.set_xlabel('Расстояние')
        ax.set_ylabel('Частота')
        ax.legend()
        
        # Box plot сравнение
        ax = axes[1, 1]
        box_data = []
        labels = []
        for name, matrix in matrices.items():
            mask = ~np.eye(matrix.shape[0], dtype=bool)
            values = matrix[mask]
            box_data.append(values)
            labels.append(name)
        
        box_plot = ax.boxplot(box_data, labels=labels, patch_artist=True)
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('Box plot распределений расстояний', fontweight='bold')
        ax.set_ylabel('Расстояние')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Анализ корреляции между метриками
        ax = axes[1, 2]
        # Создаем матрицу корреляции между метриками
        metric_vectors = {}
        for name, matrix in matrices.items():
            mask = ~np.eye(matrix.shape[0], dtype=bool)
            metric_vectors[name] = matrix[mask]
        
        corr_matrix = np.zeros((len(matrices), len(matrices)))
        metric_names = list(matrices.keys())
        
        for i, name1 in enumerate(metric_names):
            for j, name2 in enumerate(metric_names):
                corr = np.corrcoef(metric_vectors[name1], metric_vectors[name2])[0, 1]
                corr_matrix[i, j] = corr
        
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax.set_title('Корреляция между метриками', fontweight='bold')
        ax.set_xticks(range(len(metric_names)))
        ax.set_yticks(range(len(metric_names)))
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.set_yticklabels(metric_names)
        
        # Добавляем аннотации
        for i in range(len(metric_names)):
            for j in range(len(metric_names)):
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                       ha='center', va='center', fontweight='bold',
                       color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
        
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig("metrics.png")
        
        # Детальный анализ и рекомендация
        print("\n" + "="*70)
        print("ДЕТАЛЬНЫЙ АНАЛИЗ МЕТРИК ДЛЯ ЧИСЛОВЫХ ПРИЗНАКОВ")
        print("="*70)
        
        # Критерии оценки:
        criteria_weights = {
            'std_dev': 0.3,      # Дифференциация (чем выше, тем лучше)
            'mean_balance': 0.25, # Баланс среднего (близко к 0.5)
            'range': 0.2,        # Широта диапазона
            'iqr': 0.15,         # Интерквартильный размах
            'stability': 0.1     # Стабильность (низкий коэффициент вариации)
        }
        
        best_metric = None
        best_score = -1
        detailed_scores = []
        
        for stats in metrics_stats:
            # Вычисляем отдельные показатели
            std_score = stats['Станд. отклонение']  # чем больше, тем лучше
            mean_balance_score = 1 - abs(stats['Среднее'] - 0.5)  # идеально 0.5
            range_score = stats['Диапазон']  # чем больше, тем лучше
            iqr_score = stats['Q3'] - stats['Q1']  # интерквартильный размах
            cv = stats['Станд. отклонение'] / stats['Среднее'] if stats['Среднее'] != 0 else 0
            stability_score = 1 - min(cv, 1)  # коэффициент вариации (чем меньше, тем стабильнее)
            
            # Итоговый балл
            total_score = (std_score * criteria_weights['std_dev'] +
                         mean_balance_score * criteria_weights['mean_balance'] +
                         range_score * criteria_weights['range'] +
                         iqr_score * criteria_weights['iqr'] +
                         stability_score * criteria_weights['stability'])
            
            detailed_scores.append({
                'Метрика': stats['Метрика'],
                'Общий балл': total_score,
                'Дифференциация (std)': std_score,
                'Баланс (mean)': mean_balance_score,
                'Диапазон': range_score,
                'IQR': iqr_score,
                'Стабильность': stability_score
            })
            
            print(f"\n{stats['Метрика']}:")
            print(f"  Среднее: {stats['Среднее']:.3f}")
            print(f"  Станд. отклонение: {stats['Станд. отклонение']:.3f} (дифференциация)")
            print(f"  Диапазон: [{stats['Min']:.3f}, {stats['Max']:.3f}]")
            print(f"  IQR: {stats['Q3'] - stats['Q1']:.3f}")
            print(f"  Общий балл: {total_score:.3f}")
            
            if total_score > best_score:
                best_score = total_score
                best_metric = stats['Метрика']
        
        # Создаем DataFrame для визуализации оценок
        scores_df = pd.DataFrame(detailed_scores)
        
        # Визуализация детальных оценок
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metrics = scores_df['Метрика']
        criteria = ['Дифференциация (std)', 'Баланс (mean)', 'Диапазон', 'IQR', 'Стабильность']
        
        x = np.arange(len(metrics))
        width = 0.15
        
        for i, criterion in enumerate(criteria):
            values = scores_df[criterion].values
            ax.bar(x + i * width, values, width, label=criterion, alpha=0.8)
        
        ax.set_xlabel('Метрики расстояния')
        ax.set_ylabel('Оценка')
        ax.set_title('Детальный анализ критериев для каждой метрики', fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig("m2.png")
        
        print(f"\n{'='*70}")
        print(f"РЕКОМЕНДУЕМАЯ МЕТРИКА: {best_metric}")
        print(f"ОБЩИЙ БАЛЛ: {best_score:.3f}")
        print(f"{'='*70}")
        
        # Обоснование выбора
        best_stats = next(stats for stats in metrics_stats if stats['Метрика'] == best_metric)
        print(f"\nОБОСНОВАНИЕ ВЫБОРА:")
        print(f"- Хорошая дифференциация (std: {best_stats['Станд. отклонение']:.3f})")
        print(f"- Сбалансированное среднее значение (mean: {best_stats['Среднее']:.3f})")
        print(f"- Широкий диапазон значений (range: {best_stats['Диапазон']:.3f})")
        
        return best_metric, matrices[best_metric], matrices

# Запуск сравнения
if __name__ == "__main__":
    print("Сравнение метрик расстояния для ЧИСЛОВЫХ признаков")
    print("=" * 60)

    comparator = NumericalDistanceComparator(df)
    best_metric, best_matrix, all_matrices = comparator.evaluate_numerical_metrics()

    # СОЗДАЕМ ОТДЕЛЬНЫЙ ГРАФИК ДЛЯ ВСЕХ 4 МЕТРИК
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Сравнение всех метрик расстояния для всего датасета (30 книг)',
                 fontsize=16, fontweight='bold')

    # Тепловые карты для всех 4 метрик
    for idx, (name, matrix) in enumerate(all_matrices.items()):
        ax = axes[idx // 2, idx % 2]

        # Создаем тепловую карту
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')

        # Настраиваем отображение индексов книг (каждую 5-ю)
        n_books = len(df)
        tick_positions = np.arange(0, n_books, 5)
        tick_labels = [f'{i}' for i in tick_positions]

        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)

        ax.set_title(f'{name}\nМатрица расстояний', fontweight='bold', fontsize=12)
        ax.set_xlabel('Индекс книги')
        ax.set_ylabel('Индекс книги')

        # Добавляем colorbar для каждой тепловой карты
        plt.colorbar(im, ax=ax, shrink=0.8, label='Расстояние')

    plt.tight_layout()
    plt.savefig("all_metrics_comparison.png", dpi=300, bbox_inches='tight')
    print("График всех метрик сохранен как 'all_metrics_comparison.png'")

    # ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ: СРАВНЕНИЕ РАСПРЕДЕЛЕНИЙ
    plt.figure(figsize=(15, 5))

    # 1. Сравнение распределений всех метрик
    plt.subplot(1, 3, 1)
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']

    for idx, (name, matrix) in enumerate(all_matrices.items()):
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        values = matrix[mask]
        plt.hist(values, bins=20, alpha=0.7, label=name, color=colors[idx], density=True)

    plt.title('Распределения всех метрик расстояния', fontweight='bold')
    plt.xlabel('Расстояние')
    plt.ylabel('Плотность вероятности')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Box plot всех метрик
    plt.subplot(1, 3, 2)
    box_data = []
    labels = []

    for name, matrix in all_matrices.items():
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        values = matrix[mask]
        box_data.append(values)
        labels.append(name)

    box_plot = plt.boxplot(box_data, labels=labels, patch_artist=True)

    # Раскрашиваем box plot
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.title('Box plot всех метрик', fontweight='bold')
    plt.ylabel('Расстояние')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # 3. Анализ вклада признаков
    plt.subplot(1, 3, 3)

    # Анализируем вклад каждого признака
    features = ['year', 'pages']
    feature_contributions = []

    for i in range(len(comparator.df)):
        for j in range(i+1, len(comparator.df)):
            vec1 = comparator.get_numerical_vector(i, normalized=False)
            vec2 = comparator.get_numerical_vector(j, normalized=False)

            # Вклад каждого признака в евклидово расстояние
            contributions = (vec1 - vec2) ** 2
            feature_contributions.append(contributions)

    feature_contributions = np.array(feature_contributions)
    mean_contributions = np.mean(feature_contributions, axis=0)
    total = np.sum(mean_contributions)
    percentages = (mean_contributions / total) * 100

    # Столбчатая диаграмма для 2 признаков
    bars = plt.bar(features, percentages, color=['#ff6b6b', '#4ecdc4'], alpha=0.8)
    plt.title('Вклад признаков в евклидово расстояние', fontweight='bold')
    plt.ylabel('Процент вклада (%)')
    plt.ylim(0, 100)

    # Добавляем значения на столбцы
    for bar, percentage in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("metrics_analysis.png", dpi=300, bbox_inches='tight')
    print("Анализ метрик сохранен как 'metrics_analysis.png'")

    # ВЫВОД СТАТИСТИКИ ПО ВСЕМ МЕТРИКАМ
    print(f"\n{'='*70}")
    print("СТАТИСТИКА ПО ВСЕМ МЕТРИКАМ ДЛЯ ВСЕГО ДАТАСЕТА (30 книг)")
    print(f"{'='*70}")

    for name, matrix in all_matrices.items():
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        values = matrix[mask]

        print(f"\n{name}:")
        print(f"  Среднее расстояние: {np.mean(values):.4f}")
        print(f"  Стандартное отклонение: {np.std(values):.4f}")
        print(f"  Медиана: {np.median(values):.4f}")
        print(f"  Минимальное: {np.min(values):.4f}")
        print(f"  Максимальное: {np.max(values):.4f}")
        print(f"  Диапазон: {np.max(values) - np.min(values):.4f}")

    print(f"\nЛУЧШАЯ МЕТРИКА: {best_metric}")
    print(f"{'='*70}")

    # АНАЛИЗ ПРИМЕРОВ ДЛЯ ЛУЧШЕЙ МЕТРИКИ
    print(f"\nПРИМЕРЫ РАССТОЯНИЙ ДЛЯ ЛУЧШЕЙ МЕТРИКИ ({best_metric}):")
    print("-" * 50)

    # Находим несколько примеров пар книг
    n_books = len(df)
    examples = []

    # Берем несколько разных пар для демонстрации
    pairs_to_show = [(0, 1), (5, 10), (15, 20), (25, 29), (2, 28)]

    for i, j in pairs_to_show:
        if i < n_books and j < n_books:
            distance = best_matrix[i, j]
            book1 = df.iloc[i]
            book2 = df.iloc[j]

            examples.append((i, j, distance, book1, book2))

    for idx, (i, j, dist, book1, book2) in enumerate(examples):
        print(f"\nПример {idx + 1}:")
        print(f"  Книга {i}: '{book1['title']}' ({book1['year']}г, {book1['pages']}стр)")
        print(f"  Книга {j}: '{book2['title']}' ({book2['year']}г, {book2['pages']}стр)")
        print(f"  Расстояние: {dist:.4f}")
        print(f"  Разница в годе: {abs(book1['year'] - book2['year'])} лет")
        print(f"  Разница в страницах: {abs(book1['pages'] - book2['pages'])} стр")
