import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загружаем данные
data_path = r'C:/Users/vtako/Desktop/Project/Homework 1/Выгрузка ЦИАН.csv'
data = pd.read_csv(data_path)

# Убираем пропуски и вычисляем цену за квадратный метр
clean_data = data.dropna(subset=['price', 'total_meters'])
clean_data = clean_data[clean_data['total_meters'] > 0]
clean_data['price_per_sqm'] = clean_data['price'] / clean_data['total_meters']

# Считаем среднюю цену за квадратный метр по районам
average_price_sqm = clean_data.groupby('district')['price_per_sqm'].mean().reset_index()

# Визуализация средней цены за квадратный метр по районам
plt.figure(figsize=(12, 6))
sns.barplot(x='district', y='price_per_sqm', data=average_price_sqm)
plt.title('Средние цены за квадратный метр по районам')
plt.xlabel('Район')
plt.ylabel('Цена за квадратный метр (руб)')
plt.xticks(rotation=90)
plt.show()

# Считаем объемы вводимого жилья по районам
housing_supply = data.groupby('district')['total_meters'].sum().reset_index()

# Визуализация объемов вводимого жилья по районам
plt.figure(figsize=(14, 7))
sns.barplot(x='district', y='total_meters', data=housing_supply)
plt.title('Объемы вводимого жилья по районам')
plt.xlabel('Район')
plt.ylabel('Объем вводимого жилья (кв.м)')
plt.xticks(rotation=90)
plt.show()

# Отбираем данные для анализа
relevant_columns = ['district', 'floors_count', 'rooms_count', 'total_meters', 'price_per_sqm']
analysis_data = clean_data[relevant_columns].dropna()

# Визуализация различных показателей
fig, axes = plt.subplots(2, 2, figsize=(18, 18))

# Этажность домов
sns.boxplot(x='district', y='floors_count', data=analysis_data, ax=axes[0, 0])
axes[0, 0].set_title('Этажность домов по районам')
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=90)

# Количество комнат
sns.boxplot(x='district', y='rooms_count', data=analysis_data, ax=axes[0, 1])
axes[0, 1].set_title('Количество комнат по районам')
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=90)

# Жилая площадь
sns.boxplot(x='district', y='total_meters', data=analysis_data, ax=axes[1, 0])
axes[1, 0].set_title('Жилая площадь по районам')
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=90)

# Цена за квадратный метр
sns.boxplot(x='district', y='price_per_sqm', data=analysis_data, ax=axes[1, 1])
axes[1, 1].set_title('Цена за квадратный метр по районам')
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=90)

plt.tight_layout()
plt.show()

# Визуализация связи между этажностью и ценой за квадратный метр
plt.figure(figsize=(12, 6))
sns.scatterplot(x='floors_count', y='price_per_sqm', hue='district', data=analysis_data)
plt.title('Связь между этажностью и ценой за квадратный метр')
plt.xlabel('Этажность')
plt.ylabel('Цена за квадратный метр (руб)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Распределение цен за квадратный метр по районам
plt.figure(figsize=(12, 6))
sns.violinplot(x='district', y='price_per_sqm', data=analysis_data)
plt.title('Распределение цен за квадратный метр по районам')
plt.xlabel('Район')
plt.ylabel('Цена за квадратный метр (руб)')
plt.xticks(rotation=90)
plt.show()

