import numpy as np

# Параметры эксперимента
total_parts = 10
non_standard_parts = 3
standard_parts = total_parts - non_standard_parts
num_samples = 100000

# Симуляция проверки деталей
def check_parts():
    parts = ['нестандартная'] * non_standard_parts + ['стандартная'] * standard_parts
    np.random.shuffle(parts)
    for i, part in enumerate(parts):
        if part == 'стандартная':
            return i + 1  # возвращаем количество проверенных деталей

# Подсчет случаев, когда было проверено ровно 2 детали
successful_outcomes = sum(1 for _ in range(num_samples) if check_parts() == 2)

# Экспериментальная вероятность
experimental_probability_parts = successful_outcomes / num_samples

# Сохранение результата в файл
output_file_path = 'C:/Users/vtako/Desktop/Project/Homework 4/Экспериментальная вероятность, что мастер проверит ровно две детали.txt'
with open(output_file_path, 'w') as file:
    file.write(f"Экспериментальная вероятность, что мастер проверит ровно две детали: {experimental_probability_parts}")

print(f"Результат сохранен в файл: {output_file_path}")
