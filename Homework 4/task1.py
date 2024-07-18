import pandas as pd
import re

# Загрузка данных из файла
file_path = 'C:/Users/vtako/Desktop/Project/Homework 4/Химический анализ родника в Нескучном саду.csv'
df = pd.read_csv(file_path, sep=';')

# Функция для проверки соответствия значения нормативу
def check_norm(value, norm):
    if 'не более' in norm:
        limit = float(re.findall(r"[-+]?\d*\.\d+|\d+", norm)[0])
        return value <= limit, f"не более {limit}"
    elif 'в пределах' in norm:
        limits = re.findall(r"[-+]?\d*\.\d+|\d+", norm)
        lower_limit = float(limits[0])
        upper_limit = float(limits[1])
        return lower_limit <= value <= upper_limit, f"в пределах {lower_limit}-{upper_limit}"
    else:
        return False, "норматив неизвестен"

# Преобразование столбца "Результат анализа" к числовому типу там, где это возможно
df['Результат анализа'] = pd.to_numeric(df['Результат анализа'], errors='coerce')

# Создание столбца с выводами
df['Вывод'] = df.apply(lambda row: f"Результат анализа {row['Результат анализа']} {row['Единица измерений']} при нормативе {row['Норматив']}. " +
                                       ("Т.о. результат анализа в норме." if check_norm(row['Результат анализа'], row['Норматив'])[0] else "Т.о. результат анализа не в норме."),
                       axis=1)

# Установка столбца "Показатель" в качестве индексного
df.set_index('Показатель', inplace=True)

# Просмотр итогового DataFrame
print(df)

# Сохранение итогового DataFrame в файл Excel
output_file_path = 'C:/Users/vtako/Desktop/Project/Homework 4/ВЫВОДЫ Химический анализ родника в Нескучном саду.xlsx'
df.to_excel(output_file_path)
