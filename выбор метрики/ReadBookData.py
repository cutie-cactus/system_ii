import pandas as pd

# Загрузка данных
df = pd.read_csv('data_books.csv')

# Проверка типов данных
print(df.dtypes)

# Преобразование типов (если нужно)
df['has_illustrations'] = df['has_illustrations'].astype(int)
df['age_restriction'] = df['age_restriction'].astype(int)
df['year'] = df['year'].astype(int)
df['pages'] = df['pages'].astype(int)

# Категориальные признаки можно преобразовать
categorical_columns = ['language', 'genre', 'publisher']
for col in categorical_columns:
    df[col] = df[col].astype('category')

print(df.head())
print(df.info())