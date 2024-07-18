import os
import cianparser
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

class RealEstateData:
    def __init__(self, location, pages=50):
        self.location = location
        self.pages = pages
        self.data = None
        self.df = None

    def fetch_data(self):
        parser = cianparser.CianParser(location=self.location)
        self.data = parser.get_flats(deal_type="sale", rooms=(1, 2, 3, 4, 5), 
                                     with_saving_csv=False, 
                                     additional_settings={"start_page": 1, "end_page": self.pages})

    def create_dataframe(self):
        self.df = pd.DataFrame(self.data)

    def save_data(self):
        csv_file = 'Выгрузка ЦИАН.csv'
        self.df.to_csv(csv_file, index=False)
        print(f"Сохранено в {csv_file}")

        excel_file = 'Выгрузка ЦИАН.xlsx'
        self.df.to_excel(excel_file, index=False)
        print(f"Сохранено в {excel_file}")

        pickle_file = 'Выгрузка ЦИАН.pkl'
        self.df.to_pickle(pickle_file)
        print(f"Сохранено в {pickle_file}")

        sqlite_file = 'Выгрузка ЦИАН.db'
        conn = sqlite3.connect(sqlite_file)
        self.df.to_sql('real_estate', conn, if_exists='replace', index=False)
        conn.close()
        print(f"Сохранено в {sqlite_file} (база данных SQLite)")

    def load_csv(self, path):
        self.df = pd.read_csv(path)
    
    def clean_data(self):
        self.df = self.df.dropna(subset=['price', 'total_meters'])
        self.df = self.df[self.df['total_meters'] > 0]
        self.df['price_per_sqm'] = self.df['price'] / self.df['total_meters']


class RealEstateAnalysis:
    def __init__(self, df):
        self.df = df

    def average_price_per_sqm(self):
        average_price_sqm = self.df.groupby('district')['price_per_sqm'].mean().reset_index()
        plt.figure(figsize=(12, 6))
        sns.barplot(x='district', y='price_per_sqm', data=average_price_sqm)
        plt.title('Средние цены за квадратный метр по районам')
        plt.xlabel('Район')
        plt.ylabel('Цена за квадратный метр (руб)')
        plt.xticks(rotation=90)
        plt.savefig('Средние цены за квадратный метр по районам.png', bbox_inches='tight')
        plt.show()

    def housing_supply(self):
        housing_supply = self.df.groupby('district')['total_meters'].sum().reset_index()
        plt.figure(figsize=(14, 7))
        sns.barplot(x='district', y='total_meters', data=housing_supply)
        plt.title('Объемы вводимого жилья по районам')
        plt.xlabel('Район')
        plt.ylabel('Объем вводимого жилья (кв.м)')
        plt.xticks(rotation=90)
        plt.savefig('Объемы вводимого жилья по районам.png', bbox_inches='tight')
        plt.show()

    def detailed_analysis(self):
        relevant_columns = ['district', 'floors_count', 'rooms_count', 'total_meters', 'price_per_sqm']
        analysis_data = self.df[relevant_columns].dropna()

        fig, axes = plt.subplots(2, 2, figsize=(18, 18))

        sns.boxplot(x='district', y='floors_count', data=analysis_data, ax=axes[0, 0])
        axes[0, 0].set_title('Этажность домов по районам')
        axes[0, 0].set_xticks(range(len(axes[0, 0].get_xticklabels())))
        axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=90)

        sns.boxplot(x='district', y='rooms_count', data=analysis_data, ax=axes[0, 1])
        axes[0, 1].set_title('Количество комнат по районам')
        axes[0, 1].set_xticks(range(len(axes[0, 1].get_xticklabels())))
        axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=90)

        sns.boxplot(x='district', y='total_meters', data=analysis_data, ax=axes[1, 0])
        axes[1, 0].set_title('Жилая площадь по районам')
        axes[1, 0].set_xticks(range(len(axes[1, 0].get_xticklabels())))
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=90)

        sns.boxplot(x='district', y='price_per_sqm', data=analysis_data, ax=axes[1, 1])
        axes[1, 1].set_title('Цена за квадратный метр по районам')
        axes[1, 1].set_xticks(range(len(axes[1, 1].get_xticklabels())))
        axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=90)

        plt.tight_layout()
        plt.savefig('Детальный анализ.png', bbox_inches='tight')
        plt.show()

    def floor_price_relation(self):
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='floors_count', y='price_per_sqm', hue='district', data=self.df)
        plt.title('Связь между этажностью и ценой за квадратный метр')
        plt.xlabel('Этажность')
        plt.ylabel('Цена за квадратный метр (руб)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig('Связь между этажностью и ценой за квадратный метр.png', bbox_inches='tight')
        plt.show()

    def price_distribution(self):
        plt.figure(figsize=(12, 6))
        sns.violinplot(x='district', y='price_per_sqm', data=self.df)
        plt.title('Распределение цен за квадратный метр по районам')
        plt.xlabel('Район')
        plt.ylabel('Цена за квадратный метр (руб)')
        plt.xticks(rotation=90)
        plt.savefig('Распределение цен за квадратный метр по районам.png', bbox_inches='tight')
        plt.show()

def main():
    # Задание 1
    real_estate = RealEstateData(location="Санкт-Петербург", pages=50)
    real_estate.fetch_data()
    real_estate.create_dataframe()
    real_estate.save_data()

    # Задание 2
    data_path = 'Выгрузка ЦИАН.csv'
    real_estate.load_csv(data_path)
    real_estate.clean_data()

    analysis = RealEstateAnalysis(real_estate.df)
    analysis.average_price_per_sqm()
    analysis.housing_supply()
    analysis.detailed_analysis()
    analysis.floor_price_relation()
    analysis.price_distribution()

if __name__ == "__main__":
    main()
