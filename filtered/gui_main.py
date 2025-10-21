from BookDistance import BookDistanceMetrics
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

class BookRecommendationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Система рекомендаций книг")
        self.root.geometry("1200x800")
        
        # Загрузка данных
        self.df = pd.read_csv('../DataBooks.csv')
        self.filtered_df = self.df.copy()
        self.metrics_full = BookDistanceMetrics(self.df)  # Метрики для полного датасета
        self.metrics_filtered = None  # Метрики для отфильтрованного датасета
        
        self.setup_gui()
        # Заполняем списки лайков/дизлайков при инициализации
        self.update_recommendation_lists()
    
    def show_statistics(self):
        # Создаем окно со статистикой
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Статистика данных")
        stats_window.geometry("600x400")
        
        # Создаем графики
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Распределение по жанрам
        genre_counts = self.filtered_df['genre'].value_counts()
        axes[0, 0].pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Распределение по жанрам')
        
        # Распределение по годам
        year_counts = self.filtered_df['year'].value_counts().sort_index()
        axes[0, 1].bar(year_counts.index, year_counts.values)
        axes[0, 1].set_title('Распределение по годам')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Распределение по количеству страниц
        axes[1, 0].hist(self.filtered_df['pages'], bins=10, alpha=0.7)
        axes[1, 0].set_title('Распределение по страницам')
        axes[1, 0].set_xlabel('Количество страниц')
        axes[1, 0].set_ylabel('Частота')
        
        # Топ авторов
        author_counts = self.filtered_df['author'].value_counts().head(10)
        axes[1, 1].barh(range(len(author_counts)), author_counts.values)
        axes[1, 1].set_yticks(range(len(author_counts)))
        axes[1, 1].set_yticklabels(author_counts.index)
        axes[1, 1].set_title('Топ авторов')
        
        plt.tight_layout()
        
        # Встраиваем график в Tkinter
        canvas = FigureCanvasTkAgg(fig, master=stats_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    
    def setup_gui(self):
        # Создаем вкладки
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Вкладка фильтрации
        self.filter_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.filter_frame, text="Фильтрация данных")
        
        # Вкладка рекомендаций
        self.recommendation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.recommendation_frame, text="Рекомендации")
        
        self.setup_filter_tab()
        self.setup_recommendation_tab()
    
    def setup_filter_tab(self):
        # Жанры (множественный выбор)
        ttk.Label(self.filter_frame, text="Жанры:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.genre_listbox = tk.Listbox(self.filter_frame, selectmode=tk.MULTIPLE, height=4, exportselection=False)
        genres = sorted(self.df['genre'].unique().tolist())
        for genre in genres:
            self.genre_listbox.insert(tk.END, genre)
        self.genre_listbox.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        
        # Год издания
        ttk.Label(self.filter_frame, text="Год издания от:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.year_min_var = tk.StringVar(value=str(self.df['year'].min()))
        self.year_min_entry = ttk.Entry(self.filter_frame, textvariable=self.year_min_var)
        self.year_min_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        
        ttk.Label(self.filter_frame, text="до:").grid(row=1, column=2, sticky='w', padx=5, pady=5)
        self.year_max_var = tk.StringVar(value=str(self.df['year'].max()))
        self.year_max_entry = ttk.Entry(self.filter_frame, textvariable=self.year_max_var)
        self.year_max_entry.grid(row=1, column=3, sticky='ew', padx=5, pady=5)
        
        # Количество страниц
        ttk.Label(self.filter_frame, text="Страницы от:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.pages_min_var = tk.StringVar(value=str(self.df['pages'].min()))
        self.pages_min_entry = ttk.Entry(self.filter_frame, textvariable=self.pages_min_var)
        self.pages_min_entry.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        
        ttk.Label(self.filter_frame, text="до:").grid(row=2, column=2, sticky='w', padx=5, pady=5)
        self.pages_max_var = tk.StringVar(value=str(self.df['pages'].max()))
        self.pages_max_entry = ttk.Entry(self.filter_frame, textvariable=self.pages_max_var)
        self.pages_max_entry.grid(row=2, column=3, sticky='ew', padx=5, pady=5)
        
        # Авторы (множественный выбор)
        ttk.Label(self.filter_frame, text="Авторы:").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.author_listbox = tk.Listbox(self.filter_frame, selectmode=tk.MULTIPLE, height=4, exportselection=False)
        authors = sorted(self.df['author'].unique().tolist())
        for author in authors:
            self.author_listbox.insert(tk.END, author)
        self.author_listbox.grid(row=3, column=1, sticky='ew', padx=5, pady=5)

        # Издательство (множественный выбор)
        ttk.Label(self.filter_frame, text="Издательства:").grid(row=4, column=0, sticky='w', padx=5, pady=5)
        publisher_frame = ttk.Frame(self.filter_frame)
        publisher_frame.grid(row=4, column=1, sticky='ew', padx=5, pady=5)
        
        self.publisher_listbox = tk.Listbox(publisher_frame, selectmode=tk.MULTIPLE, height=4, exportselection=False)
        publishers = sorted(self.df['publisher'].unique().tolist())
        for publisher in publishers:
            self.publisher_listbox.insert(tk.END, publisher)
        
        # Добавляем скроллбар для списка издательств
        publisher_scrollbar = ttk.Scrollbar(publisher_frame, orient=tk.VERTICAL, command=self.publisher_listbox.yview)
        self.publisher_listbox.configure(yscrollcommand=publisher_scrollbar.set)
        
        self.publisher_listbox.pack(side='left', fill='both', expand=True)
        publisher_scrollbar.pack(side='right', fill='y')

        # Язык
        ttk.Label(self.filter_frame, text="Язык:").grid(row=5, column=0, sticky='w', padx=5, pady=5)
        self.language_var = tk.StringVar(value="Все")
        languages = ["Все"] + sorted(self.df['language'].unique().tolist())
        self.language_combo = ttk.Combobox(self.filter_frame, textvariable=self.language_var, values=languages, state="readonly")
        self.language_combo.grid(row=5, column=1, sticky='ew', padx=5, pady=5)

        # Возрастное ограничение
        ttk.Label(self.filter_frame, text="Возрастное ограничение:").grid(row=6, column=0, sticky='w', padx=5, pady=5)
        self.age_var = tk.StringVar(value="Все")
        ages = ["Все"] + sorted(self.df['age_restriction'].unique().tolist())
        self.age_combo = ttk.Combobox(self.filter_frame, textvariable=self.age_var, values=ages, state="readonly")
        self.age_combo.grid(row=6, column=1, sticky='ew', padx=5, pady=5)

        # Наличие иллюстраций
        ttk.Label(self.filter_frame, text="Иллюстрации:").grid(row=7, column=0, sticky='w', padx=5, pady=5)
        self.illustrations_var = tk.StringVar(value="Все")
        illustrations = ["Все", "Есть", "Нет"]
        self.illustrations_combo = ttk.Combobox(self.filter_frame, textvariable=self.illustrations_var, values=illustrations, state="readonly")
        self.illustrations_combo.grid(row=7, column=1, sticky='ew', padx=5, pady=5)

        # Название (поиск)
        ttk.Label(self.filter_frame, text="Название (поиск):").grid(row=8, column=0, sticky='w', padx=5, pady=5)
        self.title_var = tk.StringVar()
        self.title_entry = ttk.Entry(self.filter_frame, textvariable=self.title_var)
        self.title_entry.grid(row=8, column=1, sticky='ew', padx=5, pady=5)

        # Кнопки
        button_frame = ttk.Frame(self.filter_frame)
        button_frame.grid(row=9, column=0, columnspan=4, pady=10)
        
        ttk.Button(button_frame, text="Применить фильтры", command=self.apply_filters).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Сбросить фильтры", command=self.reset_filters).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Показать статистику", command=self.show_statistics).pack(side='left', padx=5)
        
        # Таблица с отфильтрованными данными
        ttk.Label(self.filter_frame, text="Отфильтрованные книги:").grid(row=10, column=0, sticky='w', padx=5, pady=5)
        
        # Создаем Treeview для отображения книг
        columns = ('title', 'author', 'publisher', 'year', 'language', 'age_restriction', 'genre', 'pages', 'has_illustrations')
        self.tree = ttk.Treeview(self.filter_frame, columns=columns, show='headings', height=15)

        # Определяем заголовки для всех колонок
        self.tree.heading('title', text='Название')
        self.tree.heading('author', text='Автор')
        self.tree.heading('publisher', text='Издательство')
        self.tree.heading('year', text='Год')
        self.tree.heading('language', text='Язык')
        self.tree.heading('age_restriction', text='Возраст')
        self.tree.heading('genre', text='Жанр')
        self.tree.heading('pages', text='Страницы')
        self.tree.heading('has_illustrations', text='Иллюстрации')

        # Настраиваем колонки
        self.tree.column('title', width=200)
        self.tree.column('author', width=120)
        self.tree.column('publisher', width=100)
        self.tree.column('year', width=60)
        self.tree.column('language', width=80)
        self.tree.column('age_restriction', width=70)
        self.tree.column('genre', width=100)
        self.tree.column('pages', width=70)
        self.tree.column('has_illustrations', width=80)
        
        self.tree.grid(row=10, column=0, columnspan=4, sticky='nsew', padx=5, pady=5)
        
        # Scrollbar для таблицы
        scrollbar = ttk.Scrollbar(self.filter_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.grid(row=10, column=4, sticky='ns')
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Настраиваем расширяемость
        self.filter_frame.columnconfigure(1, weight=1)
        self.filter_frame.rowconfigure(10, weight=1)
        
        # Заполняем таблицу первоначальными данными
        self.update_book_table()
    
    def setup_recommendation_tab(self):
        # Левый фрейм для выбора книг
        left_frame = ttk.Frame(self.recommendation_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Правый фрейм для рекомендаций
        right_frame = ttk.Frame(self.recommendation_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        # Список книг для лайков
        ttk.Label(left_frame, text="Выберите понравившиеся книги:").pack(anchor='w', pady=(0, 5))
        
        self.likes_listbox = tk.Listbox(left_frame, selectmode=tk.MULTIPLE, height=10, exportselection=False)
        self.likes_listbox.pack(fill='both', expand=True, pady=(0, 10))
        
        # Список книг для дизлайков
        ttk.Label(left_frame, text="Выберите НЕпонравившиеся книги:").pack(anchor='w', pady=(0, 5))
        
        self.dislikes_listbox = tk.Listbox(left_frame, selectmode=tk.MULTIPLE, height=10, exportselection=False)
        self.dislikes_listbox.pack(fill='both', expand=True, pady=(0, 10))
        
        # Настройки рекомендаций
        settings_frame = ttk.Frame(left_frame)
        settings_frame.pack(fill='x', pady=10)
        
        ttk.Label(settings_frame, text="Количество рекомендаций:").grid(row=0, column=0, sticky='w')
        self.n_rec_var = tk.IntVar(value=5)
        ttk.Spinbox(settings_frame, from_=1, to=20, textvariable=self.n_rec_var, width=5).grid(row=0, column=1, padx=5)
        
        ttk.Label(settings_frame, text="Стратегия:").grid(row=1, column=0, sticky='w', pady=5)
        self.strategy_var = tk.StringVar(value="combined")
        strategies = [("Комбинированная", "combined"), 
                     ("Усреднение", "average"),
                     ("Объединение", "union"),
                     ("Контент-усиление", "content_boost")]
        
        for i, (text, value) in enumerate(strategies):
            ttk.Radiobutton(settings_frame, text=text, variable=self.strategy_var, value=value).grid(row=1, column=i+1, sticky='w', padx=5)
        
        ttk.Label(settings_frame, text="Коэффициент штрафа:").grid(row=2, column=0, sticky='w', pady=5)
        self.penalty_var = tk.DoubleVar(value=0.4)
        ttk.Scale(settings_frame, from_=0.0, to=1.0, variable=self.penalty_var, orient='horizontal').grid(row=2, column=1, columnspan=3, sticky='ew', padx=5)
        ttk.Label(settings_frame, textvariable=self.penalty_var).grid(row=2, column=4, padx=5)
        
        # Кнопка получения рекомендаций
        ttk.Button(left_frame, text="Получить рекомендации", command=self.get_recommendations).pack(pady=10)
        
        # Область для вывода рекомендаций
        ttk.Label(right_frame, text="Рекомендации:").pack(anchor='w')
        
        self.recommendation_text = scrolledtext.ScrolledText(right_frame, height=20, width=60)
        self.recommendation_text.pack(fill='both', expand=True, pady=5)
        
        # Кнопка очистки
        ttk.Button(right_frame, text="Очистить рекомендации", command=self.clear_recommendations).pack(pady=5)
        
        # Обновляем списки книг
        self.update_recommendation_lists()
    
    def apply_filters(self):
        try:
            # Применяем фильтры
            filtered_df = self.df.copy()
            
            # Фильтр по жанру (множественный выбор)
            selected_genres = [self.genre_listbox.get(i) for i in self.genre_listbox.curselection()]
            if selected_genres:
                filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]
            
            # Фильтр по авторам (множественный выбор)
            selected_authors = [self.author_listbox.get(i) for i in self.author_listbox.curselection()]
            if selected_authors:
                filtered_df = filtered_df[filtered_df['author'].isin(selected_authors)]
            
            # Фильтр по издательствам (множественный выбор)
            selected_publishers = [self.publisher_listbox.get(i) for i in self.publisher_listbox.curselection()]
            if selected_publishers:
                filtered_df = filtered_df[filtered_df['publisher'].isin(selected_publishers)]
            
            # Фильтр по языку
            if self.language_var.get() != "Все":
                filtered_df = filtered_df[filtered_df['language'] == self.language_var.get()]
            
            # Фильтр по возрастному ограничению
            if self.age_var.get() != "Все":
                age_value = int(self.age_var.get())
                filtered_df = filtered_df[filtered_df['age_restriction'] == age_value]
            
            # Фильтр по иллюстрациям
            if self.illustrations_var.get() != "Все":
                has_illustrations = 1 if self.illustrations_var.get() == "Есть" else 0
                filtered_df = filtered_df[filtered_df['has_illustrations'] == has_illustrations]
            
            # Фильтр по названию (поиск)
            if self.title_var.get():
                filtered_df = filtered_df[filtered_df['title'].str.contains(self.title_var.get(), case=False, na=False)]
            
            # Фильтр по году
            year_min = int(self.year_min_var.get()) if self.year_min_var.get() else self.df['year'].min()
            year_max = int(self.year_max_var.get()) if self.year_max_var.get() else self.df['year'].max()
            filtered_df = filtered_df[(filtered_df['year'] >= year_min) & (filtered_df['year'] <= year_max)]
            
            # Фильтр по страницам
            pages_min = int(self.pages_min_var.get()) if self.pages_min_var.get() else self.df['pages'].min()
            pages_max = int(self.pages_max_var.get()) if self.pages_max_var.get() else self.df['pages'].max()
            filtered_df = filtered_df[(filtered_df['pages'] >= pages_min) & (filtered_df['pages'] <= pages_max)]
            
            self.filtered_df = filtered_df.reset_index(drop=True)
            self.update_book_table()
            
            # Создаем метрики для отфильтрованного датасета
            self.metrics_filtered = BookDistanceMetrics(self.filtered_df)
            
            messagebox.showinfo("Успех", f"Найдено {len(self.filtered_df)} книг")
            
        except ValueError as e:
            messagebox.showerror("Ошибка", "Проверьте правильность введенных значений")
    
    def reset_filters(self):
        # Сброс множественного выбора жанров, авторов и издательств
        self.genre_listbox.selection_clear(0, tk.END)
        self.author_listbox.selection_clear(0, tk.END)
        self.publisher_listbox.selection_clear(0, tk.END)
        
        # Сброс остальных фильтров
        self.language_var.set("Все")
        self.age_var.set("Все")
        self.illustrations_var.set("Все")
        self.title_var.set("")
        self.year_min_var.set(str(self.df['year'].min()))
        self.year_max_var.set(str(self.df['year'].max()))
        self.pages_min_var.set(str(self.df['pages'].min()))
        self.pages_max_var.set(str(self.df['pages'].max()))
        
        self.filtered_df = self.df.copy()
        self.update_book_table()
        
        # Создаем метрики для отфильтрованного датасета
        self.metrics_filtered = BookDistanceMetrics(self.filtered_df)
        
    def update_book_table(self):
        # Очищаем таблицу
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Заполняем таблицу отфильтрованными данными со всеми колонками
        for idx, row in self.filtered_df.iterrows():
            illustrations = "Есть" if row['has_illustrations'] == 1 else "Нет"
            self.tree.insert('', 'end', values=(
                row['title'],
                row['author'],
                row['publisher'],
                row['year'],
                row['language'],
                row['age_restriction'],
                row['genre'],
                row['pages'],
                illustrations
            ))
        
    def update_recommendation_lists(self):
        # Очищаем списки
        self.likes_listbox.delete(0, tk.END)
        self.dislikes_listbox.delete(0, tk.END)
        
        # Заполняем списки названиями книг из ПОЛНОГО датасета
        for idx, row in self.df.iterrows():
            book_text = f"{row['title']} - {row['author']}"
            self.likes_listbox.insert(tk.END, book_text)
            self.dislikes_listbox.insert(tk.END, book_text)
    
    def get_recommendations(self):
        if not hasattr(self, 'filtered_df') or len(self.filtered_df) == 0:
            messagebox.showwarning("Предупреждение", "Нет данных для рекомендаций")
            return
        
        try:
            # Получаем выбранные лайки и дизлайки из ВСЕГО датасета
            liked_indices_full = list(self.likes_listbox.curselection())
            disliked_indices_full = list(self.dislikes_listbox.curselection())
            
            if not liked_indices_full:
                messagebox.showwarning("Предупреждение", "Выберите хотя бы одну понравившуюся книгу")
                return
            
            # Создаем метрики для отфильтрованного датасета, если еще не созданы
            if self.metrics_filtered is None:
                self.metrics_filtered = BookDistanceMetrics(self.filtered_df)
            
            # Получаем рекомендации
            recommendations = recommend_based_on_multiple_likes(
                self.metrics_full,  # Для лайков/дизлайков используем полный датасет
                self.metrics_filtered,  # Для рекомендаций используем отфильтрованный датасет
                liked_indices_full,
                n_recommendations=self.n_rec_var.get(),
                strategy=self.strategy_var.get(),
                disliked_book_indices=disliked_indices_full,
                penalty_factor=self.penalty_var.get(),
                verbose=False
            )
            
            # Форматируем вывод рекомендаций
            self.display_recommendations(recommendations, liked_indices_full, disliked_indices_full)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")
    
    def display_recommendations(self, recommendations, liked_indices_full, disliked_indices_full):
        # Очищаем текстовое поле
        self.recommendation_text.delete(1.0, tk.END)
        
        # Выводим информацию о выбранных книгах из полного датасета
        self.recommendation_text.insert(tk.END, "=" * 70 + "\n")
        self.recommendation_text.insert(tk.END, "ВАШИ ПРЕДПОЧТЕНИЯ:\n")
        self.recommendation_text.insert(tk.END, "=" * 70 + "\n\n")
        
        self.recommendation_text.insert(tk.END, "👍 ПОНРАВИЛИСЬ:\n")
        for idx in liked_indices_full:
            book = self.df.iloc[idx]  # Берем из полного датасета
            self.recommendation_text.insert(tk.END, f"• {book['title']} - {book['author']}\n")
        
        if disliked_indices_full:
            self.recommendation_text.insert(tk.END, "\n👎 НЕ ПОНРАВИЛИСЬ:\n")
            for idx in disliked_indices_full:
                book = self.df.iloc[idx]  # Берем из полного датасета
                self.recommendation_text.insert(tk.END, f"• {book['title']} - {book['author']}\n")
        
        self.recommendation_text.insert(tk.END, "\n" + "=" * 70 + "\n")
        self.recommendation_text.insert(tk.END, "РЕКОМЕНДАЦИИ (из отфильтрованного датасета):\n")
        self.recommendation_text.insert(tk.END, "=" * 70 + "\n\n")
        
        if not recommendations:
            self.recommendation_text.insert(tk.END, "Не найдено подходящих рекомендаций\n")
            return
        
        # Выводим рекомендации из отфильтрованного датасета
        for i, (book_idx, similarity) in enumerate(recommendations, 1):
            book = self.filtered_df.iloc[book_idx]  # Берем из отфильтрованного датасета
            
            self.recommendation_text.insert(tk.END, f"{i}. {book['title']} - {book['author']}\n")
            self.recommendation_text.insert(tk.END, f"   Жанр: {book['genre']}, Год: {book['year']}, Страниц: {book['pages']}\n")
            self.recommendation_text.insert(tk.END, f"   Схожесть: {similarity:.3f}\n")
            
            # Находим наиболее похожую книгу из лайков (из полного датасета)
            best_match = None
            best_similarity = 0
            for liked_idx in liked_indices_full:
                sim = self.metrics_full.similarity_score(book_idx, liked_idx)
                if sim > best_similarity:
                    best_similarity = sim
                    best_match = self.df.iloc[liked_idx]  # Берем из полного датасета
            
            if best_match is not None:
                self.recommendation_text.insert(tk.END, f"   Похожа на: '{best_match['title']}' (схожесть: {best_similarity:.3f})\n")
            
            # Проверяем общие черты с лайками (из полного датасета)
            common_features = []
            for liked_idx in liked_indices_full:
                liked_book = self.df.iloc[liked_idx]  # Берем из полного датасета
                if book['genre'] == liked_book['genre']:
                    common_features.append(f"жанр '{liked_book['genre']}'")
                if book['author'] == liked_book['author']:
                    common_features.append(f"автор {liked_book['author']}")
            
            if common_features:
                self.recommendation_text.insert(tk.END, f"   Общие черты: {', '.join(set(common_features))}\n")
            
            self.recommendation_text.insert(tk.END, "\n")
    
    def clear_recommendations(self):
        self.recommendation_text.delete(1.0, tk.END)


# Функции рекомендательной системы (с исправлениями для работы с двумя датасетами)
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

def recommend_based_on_multiple_likes(metrics_full, metrics_filtered, liked_book_indices, n_recommendations=10, weights=None, 
                                     exclude_liked=True, strategy='combined', disliked_book_indices=None,
                                     penalty_factor=0.7, verbose=True):
    """
    Рекомендация книг на основе нескольких понравившихся книг
    metrics_full - для лайков/дизлайков (полный датасет)
    metrics_filtered - для рекомендаций (отфильтрованный датасет)
    """
    
    if not liked_book_indices:
        if verbose:
            print("Ошибка: список понравившихся книг пуст")
        return []
    
    # Инициализируем список дизлайков если None
    if disliked_book_indices is None:
        disliked_book_indices = []
    
    if verbose:
        print("=" * 70)
        print("РЕКОМЕНДАЦИИ НА ОСНОВЕ ВАШИХ ПРЕДПОЧТЕНИЙ:")
        print("=" * 70)
        
        print("👍 ПОНРАВИЛИСЬ:")
        for i, idx in enumerate(liked_book_indices):
            # Берем книги из полного датасета для отображения
            book = metrics_full.df.iloc[idx]
            print(f"  {i+1}. '{book['title']}' - {book['author']} ({book['genre']})")
        
        if disliked_book_indices:
            print("\n👎 НЕ ПОНРАВИЛИСЬ:")
            for i, idx in enumerate(disliked_book_indices):
                book = metrics_full.df.iloc[idx]
                print(f"  {i+1}. '{book['title']}' - {book['author']} ({book['genre']})")
        print()
    
    # Получаем оценки для ВСЕХ книг в ОТФИЛЬТРОВАННОМ датасете с учетом стратегии
    if strategy == 'combined':
        all_scores = _combined_strategy_all_books(metrics_full, metrics_filtered, liked_book_indices, weights, exclude_liked)
    elif strategy == 'average':
        all_scores = _average_strategy_all_books(metrics_full, metrics_filtered, liked_book_indices, weights, exclude_liked)
    elif strategy == 'union':
        all_scores = _union_strategy_all_books(metrics_full, metrics_filtered, liked_book_indices, weights, exclude_liked)
    elif strategy == 'content_boost':
        all_scores = _content_boost_strategy_all_books(metrics_full, metrics_filtered, liked_book_indices, weights, exclude_liked)
    else:
        all_scores = _combined_strategy_all_books(metrics_full, metrics_filtered, liked_book_indices, weights, exclude_liked)
    
    # Применяем штраф за дизлайки ко ВСЕМ книгам в ОТФИЛЬТРОВАННОМ датасете
    if disliked_book_indices:
        all_scores = _apply_dislike_penalty_all_books(metrics_full, metrics_filtered, all_scores, disliked_book_indices, penalty_factor)
    
    # Сортируем и выбираем лучшие
    all_recommendations = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    recommendations = all_recommendations[:n_recommendations]
    
    if verbose:
        _display_recommendations(metrics_full, metrics_filtered, recommendations, liked_book_indices, disliked_book_indices)
    
    return recommendations


def _apply_dislike_penalty_all_books(metrics_full, metrics_filtered, all_scores, disliked_indices, penalty_factor):
    """Применяет штраф ко ВСЕМ книгам в отфильтрованном датасете на основе дизлайков из полного датасета"""
    penalized_scores = {}
    
    for book_idx, similarity in all_scores.items():
        if book_idx in disliked_indices:
            continue  # Полностью исключаем дизлайки
            
        # Вычисляем схожесть с дизлайками (из полного датасета)
        max_dislike_similarity = 0
        for dislike_idx in disliked_indices:
            dislike_sim = metrics_full.similarity_score(book_idx, dislike_idx)
            max_dislike_similarity = max(max_dislike_similarity, dislike_sim)
        
        # Применяем штраф
        penalty = max_dislike_similarity * penalty_factor
        penalized_similarity = similarity * (1 - penalty)
        
        penalized_scores[book_idx] = max(penalized_similarity, 0)
    
    return penalized_scores


def _combined_strategy_all_books(metrics_full, metrics_filtered, liked_indices, weights, exclude_liked):
    """Комбинированная стратегия для ВСЕХ книг в отфильтрованном датасете"""
    book_scores = {}
    
    for book_idx in range(len(metrics_filtered.df)):
        if exclude_liked and book_idx in liked_indices:
            continue
            
        total_similarity = 0
        for liked_idx in liked_indices:
            similarity = metrics_full.similarity_score(book_idx, liked_idx, weights)
            total_similarity += similarity
        
        avg_similarity = total_similarity / len(liked_indices)
        book_scores[book_idx] = avg_similarity
    
    # Усиливаем рекомендации с общими признаками
    boosted_scores = _boost_by_common_features(metrics_full, metrics_filtered, liked_indices, book_scores)
    
    return boosted_scores


def _average_strategy_all_books(metrics_full, metrics_filtered, liked_indices, weights, exclude_liked):
    """Стратегия усреднения для ВСЕХ книг в отфильтрованном датасете"""
    book_scores = {}
    
    for book_idx in range(len(metrics_filtered.df)):
        if exclude_liked and book_idx in liked_indices:
            continue
            
        total_similarity = 0
        for liked_idx in liked_indices:
            similarity = metrics_full.similarity_score(book_idx, liked_idx, weights)
            total_similarity += similarity
        
        avg_similarity = total_similarity / len(liked_indices)
        book_scores[book_idx] = avg_similarity
    
    return book_scores


def _union_strategy_all_books(metrics_full, metrics_filtered, liked_indices, weights, exclude_liked):
    """Стратегия объединения для ВСЕХ книг в отфильтрованном датасете"""
    max_scores = {}
    
    for book_idx in range(len(metrics_filtered.df)):
        if exclude_liked and book_idx in liked_indices:
            continue
            
        max_similarity = 0
        for liked_idx in liked_indices:
            similarity = metrics_full.similarity_score(book_idx, liked_idx, weights)
            max_similarity = max(max_similarity, similarity)
        
        max_scores[book_idx] = max_similarity
    
    return max_scores


def _content_boost_strategy_all_books(metrics_full, metrics_filtered, liked_indices, weights, exclude_liked):
    """Стратегия усиления контента для ВСЕХ книг в отфильтрованном датасете"""
    base_scores = _average_strategy_all_books(metrics_full, metrics_filtered, liked_indices, weights, exclude_liked)
    boosted_scores = _boost_by_common_features(metrics_full, metrics_filtered, liked_indices, base_scores)
    return boosted_scores


def _boost_by_common_features(metrics_full, metrics_filtered, liked_indices, book_scores):
    """Усиление оценок на основе общих признаков с лайками из полного датасета"""
    boosted_scores = book_scores.copy()
    
    liked_books = [metrics_full.df.iloc[idx] for idx in liked_indices]
    
    genres = [book['genre'] for book in liked_books]
    authors = [book['author'] for book in liked_books]
    
    genre_counter = Counter(genres)
    author_counter = Counter(authors)
    
    most_common_genre = genre_counter.most_common(1)[0][0] if genre_counter else None
    most_common_author = author_counter.most_common(1)[0][0] if author_counter else None
    
    for book_idx in boosted_scores:
        book = metrics_filtered.df.iloc[book_idx]
        
        boost = 1.0
        
        if most_common_genre and book['genre'] == most_common_genre:
            boost *= 1.2
        
        if most_common_author and book['author'] == most_common_author:
            boost *= 1.3
        
        matching_genres = sum(1 for liked_genre in genres if liked_genre == book['genre'])
        if matching_genres > 1:
            boost *= (1 + 0.15 * matching_genres)

        matching_authors = sum(1 for liked_author in authors if liked_author == book['author'])
        if matching_authors > 1:
            boost *= (1 + 0.2 * matching_authors)
        
        boosted_scores[book_idx] *= boost
    
    return boosted_scores


def _display_recommendations(metrics_full, metrics_filtered, recommendations, liked_indices, disliked_indices=None):
    """Отображение рекомендаций с анализом"""
    if disliked_indices is None:
        disliked_indices = []
        
    print("ТОП РЕКОМЕНДАЦИЙ:")
    print("-" * 70)
    
    if not recommendations:
        print("Не найдено подходящих рекомендаций с учетом ваших предпочтений")
        return
    
    for i, (book_idx, similarity) in enumerate(recommendations, 1):
        book = metrics_filtered.df.iloc[book_idx]
        
        # Находим наиболее похожие книги из лайков (из полного датасета)
        best_matches = []
        for liked_idx in liked_indices:
            liked_book = metrics_full.df.iloc[liked_idx]
            sim = metrics_full.similarity_score(book_idx, liked_idx)
            best_matches.append((liked_book['title'], sim))
        
        best_matches.sort(key=lambda x: x[1], reverse=True)
        top_match = best_matches[0] if best_matches else ("", 0)
        
        # Проверяем схожесть с дизлайками (из полного датасета)
        max_dislike_similarity = 0
        if disliked_indices:
            for dislike_idx in disliked_indices:
                dislike_sim = metrics_full.similarity_score(book_idx, dislike_idx)
                max_dislike_similarity = max(max_dislike_similarity, dislike_sim)
        
        print(f"{i}. {book['title']} - {book['author']}")
        print(f"   Жанр: {book['genre']}, Год: {book['year']}, Страниц: {book['pages']}")
        print(f"   Общая схожесть: {similarity:.3f}")
        
        if top_match[1] > 0:
            print(f"   Наиболее похожа на: '{top_match[0]}' (схожесть: {top_match[1]:.3f})")
        
        if max_dislike_similarity > 0.6:
            print(f"   ⚠️  Умеренно похожа на непонравившиеся книги (схожесть: {max_dislike_similarity:.3f})")
        
        # Показываем общие черты с понравившимися книгами (из полного датасета)
        common_features = []
        for liked_idx in liked_indices:
            liked_book = metrics_full.df.iloc[liked_idx]
            if book['genre'] == liked_book['genre']:
                common_features.append(f"жанр '{liked_book['genre']}'")
            if book['author'] == liked_book['author']:
                common_features.append(f"автор {liked_book['author']}")
        
        if common_features:
            print(f"   ✅ Общие черты: {', '.join(set(common_features))}")
        
        print()


# Запуск приложения
if __name__ == "__main__":
    root = tk.Tk()
    app = BookRecommendationGUI(root)
    root.mainloop()