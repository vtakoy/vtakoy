import os
import ssl
import argparse

# Парсим аргументы командной строки
parser = argparse.ArgumentParser(description='Запуск анализатора документов в консоли')
parser.add_argument('--offline', action='store_true', help='Запуск в полностью офлайн-режиме')
parser.add_argument('--no-vector-db', action='store_true', help='Запуск без использования векторной базы данных')
parser.add_argument('--location', choices=['home', 'work'], help='Местоположение (дома или на работе)')
args = parser.parse_args()

# Загружаем настройки из .env
from dotenv import load_dotenv
load_dotenv()

# Проверяем, нужно ли запускать в офлайн-режиме
OFFLINE_MODE = args.offline or os.getenv('OFFLINE_MODE', 'False').lower() == 'true'
if OFFLINE_MODE:
    print("Запуск в офлайн-режиме: все сетевые запросы отключены")
    os.environ['OFFLINE_MODE'] = 'true'

# Проверяем, нужно ли отключить векторную базу данных
DISABLE_VECTOR_DB = args.no_vector_db or os.getenv('DISABLE_VECTOR_DB', 'False').lower() == 'true'
if DISABLE_VECTOR_DB:
    print("Запуск без использования векторной базы данных")
    os.environ['DISABLE_VECTOR_DB'] = 'true'
    # Разрешаем продолжение работы без векторного хранилища
    os.environ['ALLOW_NO_VECTOR_DB'] = 'true'

# Отключаем проверку SSL и телеметрию
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CHROMA_TELEMETRY_ENABLED'] = 'False'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGCHAIN_TRACING'] = 'false'
os.environ['LANGCHAIN_SESSION'] = 'false'
os.environ['LANGCHAIN_ENDPOINT'] = ''
os.environ['LANGCHAIN_API_KEY'] = ''
os.environ['LANGCHAIN_PROJECT'] = ''

# Устанавливаем местоположение из командной строки, если задано
if args.location:
    os.environ['WORK_LOCATION'] = args.location
    print(f"Установлено местоположение из командной строки: {args.location}")

# Определяем местоположение - дома или на работе
work_location = os.getenv("WORK_LOCATION", "home").lower()
print(f"Режим работы: {'на работе' if work_location == 'work' else 'дома'}")

# Принудительно запрещаем все сетевые запросы в офлайн-режиме
if OFFLINE_MODE:
    os.environ['NO_PROXY'] = '*'
    os.environ['HTTP_PROXY'] = 'http://localhost:1'
    os.environ['HTTPS_PROXY'] = 'http://localhost:1'
else:
    # Если заданы реальные прокси в .env, используем их
    http_proxy = os.getenv('HTTP_PROXY')
    https_proxy = os.getenv('HTTPS_PROXY')
    if http_proxy and https_proxy:
        print(f"Используются пользовательские настройки прокси")

ssl._create_default_https_context = ssl._create_unverified_context

from auth import GigaChatAuth
from document_analyzer import DocumentAnalyzer

def main():
    # Получаем учетные данные GigaChat из переменных окружения
    client_id = os.getenv("GIGACHAT_CLIENT_ID")
    client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("Ошибка: Не найдены учетные данные GigaChat в файле .env")
        print("Пожалуйста, создайте файл .env с переменными GIGACHAT_CLIENT_ID и GIGACHAT_CLIENT_SECRET")
        return
    
    # Инициализируем клиент GigaChat
    client = GigaChatAuth(client_id, client_secret, verify_ssl=False)
    
    # Инициализируем анализатор документов
    analyzer = DocumentAnalyzer(client)
    
    # Принудительно перечитываем документы
    print("Запуск индексации документов с улучшенными алгоритмами обработки...")
    analyzer.read_documents()
    
    print("\nДобро пожаловать в AI-ассистент для анализа документов!")
    print("Я могу анализировать следующие типы документов:")
    print("- PDF документы")
    print("- Word документы (.docx)")
    print("- Excel файлы (.xlsx, .xls)")
    print("- Текстовые файлы (.txt)")
    print("\nЧтобы задать вопрос о документах, начните запрос с 'Требуется информация'")
    print("Например: 'Требуется информация о методах обработки данных'")
    print("\nДля выхода введите 'выход' или 'exit'")
    
    while True:
        user_input = input("\nВаш запрос: ")
        
        if user_input.lower() in ["выход", "exit"]:
            print("До свидания!")
            break
        
        if user_input.lower().startswith("требуется информация"):
            response = analyzer.analyze_documents(user_input)
            print(f"\nОтвет: {response['result']}")
            
            if response['sources']:
                print("\nИсточники:")
                for source in response['sources']:
                    print(f"- {source}")
        else:
            print("Запрос должен начинаться с 'Требуется информация'")

if __name__ == "__main__":
    main() 