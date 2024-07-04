import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3

# Функция для получения HTML страницы
def get_html(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    return response.text

# Функция для парсинга HTML и извлечения данных
def parse_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    data = []
    items = soup.find_all('article', class_='a-card')
    for item in items:
        title = item.find('div', class_='a-card__title').get_text(strip=True)
        price = item.find('div', class_='a-card__price').get_text(strip=True)
        address = item.find('div', class_='a-card__subtitle').get_text(strip=True)
        details = item.find('div', class_='a-card__details').get_text(strip=True)
        data.append({
            'title': title,
            'price': price,
            'address': address,
            'details': details
        })
    return data

# Основная функция для скрапинга данных с нескольких страниц
def scrape_cian_data(pages=10):
    base_url = "https://www.cian.ru/cat.php?deal_type=sale&engine_version=2&offer_type=flat&p={}"
    all_data = []
    for page in range(1, pages + 1):
        url = base_url.format(page)
        html = get_html(url)
        data = parse_html(html)
        all_data.extend(data)
    return pd.DataFrame(all_data)

# Скрапинг данных
df = scrape_cian_data(pages=10)

# Сохранение в CSV
df.to_csv('real_estate_data.csv', index=False)

# Сохранение в Excel
df.to_excel('real_estate_data.xlsx', index=False)

# Сохранение в Pickle
df.to_pickle('real_estate_data.pkl')

# Сохранение в базу данных SQLite
conn = sqlite3.connect('real_estate_data.db')
df.to_sql('real_estate', conn, if_exists='replace', index=False)
conn.close()
