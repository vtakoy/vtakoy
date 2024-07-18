import numpy as np

# Параметры эксперимента
total_fruits = 9
total_oranges = 5
total_apples = 4
num_samples = 100000

# Симуляция выборки 3 фруктов из 9
results = np.random.choice(['апельсин'] * total_oranges + ['яблоко'] * total_apples, (num_samples, 3))

# Подсчет случаев, когда все три фрукта - апельсины
successful_outcomes = np.sum(np.all(results == 'апельсин', axis=1))

# Экспериментальная вероятность
experimental_probability_oranges = successful_outcomes / num_samples

# Сохранение результата в файл
output_file_path = 'C:/Users/vtako/Desktop/Project/Homework 4/Экспериментальная вероятность, что все три фрукта – апельсины.txt'
with open(output_file_path, 'w') as file:
    file.write(f"Экспериментальная вероятность, что все три фрукта – апельсины: {experimental_probability_oranges}")

print(f"Результат сохранен в файл: {output_file_path}")
