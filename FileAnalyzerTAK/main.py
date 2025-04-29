import os
from dotenv import load_dotenv
from auth import GigaChatAuth
from logger import logger
from gigachat_wrapper import GigaChatWrapper
from file import DocumentAnalyzer

def main():
    # Загружаем переменные окружения из .env файла
    load_dotenv()
    
    # Получаем учетные данные GigaChat из переменных окружения
    client_id = os.getenv("GIGACHAT_CLIENT_ID")
    client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("Ошибка: Не найдены учетные данные GigaChat в файле .env")
        print("Пожалуйста, создайте файл .env с переменными GIGACHAT_CLIENT_ID и GIGACHAT_CLIENT_SECRET")
        return
    
    # Инициализируем клиент GigaChat
    client = GigaChatAuth(client_id, client_secret)
    
    # Инициализируем анализатор документов
    analyzer = DocumentAnalyzer(client)
    
    # Принудительно перечитываем документы для применения улучшений
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
            print(f"\nОтвет: {response}")
        else:
            print("Запрос должен начинаться с 'Требуется информация'")

if __name__ == "__main__":
    main() 