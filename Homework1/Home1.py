import os
import cianparser
import pandas as pd
import sqlite3

# Функция для извлечения данных с ЦИАНа
def fetch_data(pages=100):
    # Создаем экземпляр парсера для ЦИАН
    parser = cianparser.CianParser(location="Санкт-Петербург")
    # Получаем данные о квартирах на продажу
    data = parser.get_flats(deal_type="sale", rooms=(1, 2, 3, 4, 5), with_saving_csv=False, additional_settings={"start_page": 1, "end_page": pages})
    return data

# Функция для создания DataFrame из данных
def create_dataframe(data):
    df = pd.DataFrame(data)
    return df

# Функция для сохранения данных в различные форматы
def save_data(df):
    # Сохранение в CSV
    csv_file = 'Выгрузка ЦИАН.csv'
    df.to_csv(csv_file, index=False)
    print(f"Сохранено в {csv_file}")

    # Сохранение в Excel
    excel_file = 'Выгрузка ЦИАН.xlsx'
    df.to_excel(excel_file, index=False)
    print(f"Сохранено в {excel_file}")

    # Сохранение в Pickle (бинарный формат для Python)
    pickle_file = 'Выгрузка ЦИАН.pkl'
    df.to_pickle(pickle_file)
    print(f"Сохранено в {pickle_file}")

    # Сохранение в базу данных SQLite
    sqlite_file = 'Выгрузка ЦИАН.db'
    conn = sqlite3.connect(sqlite_file)
    df.to_sql('real_estate', conn, if_exists='replace', index=False)
    conn.close()
    print(f"Сохранено в {sqlite_file} (база данных SQLite)")

# Основная функция программы
def main(pages=50):
    # Получаем данные
    data = fetch_data(pages)
    # Создаем DataFrame из данных
    df = create_dataframe(data)
    # Сохраняем данные
    save_data(df)

# Запуск скрипта, если он используется как отдельное приложение
if __name__ == "__main__":
    main(pages=50)
