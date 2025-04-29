import os
import sys
from dotenv import load_dotenv
from auth import GigaChatAuth
from gigachat_langchain import GigaChatLangchain
from langchain_gigachat.embeddings import GigaChatEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import urllib3
import ssl

# Отключаем предупреждения SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Загружаем переменные окружения
load_dotenv()

def demo_gigachat_embeddings():
    """Демонстрация работы с GigaChatEmbeddings через Chroma"""
    print("Демонстрация работы с GigaChatEmbeddings")
    
    # Получаем учетные данные GigaChat
    client_id = os.getenv("GIGACHAT_CLIENT_ID")
    client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("Ошибка: Не указаны GIGACHAT_CLIENT_ID или GIGACHAT_CLIENT_SECRET в .env файле")
        sys.exit(1)
    
    print("Инициализация GigaChatEmbeddings...")
    
    # Создаем экземпляр GigaChatEmbeddings
    credentials = f"{client_id}:{client_secret}"
    embeddings = GigaChatEmbeddings(
        credentials=credentials,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS",
        model="Embeddings"  # Базовая модель для эмбеддингов
    )
    
    # Создаем тестовые документы
    documents = [
        Document(page_content="Кошки — независимые животные, которым нужно собственное пространство.", 
                metadata={"source": "mammal-pets-doc"}),
        Document(page_content="Собаки — отличные компаньоны, которые известны своей преданностью и дружелюбием.", 
                metadata={"source": "mammal-pets-doc"}),
        Document(page_content="Кролики — социальные животные, которым нужно много места, чтобы прыгать.", 
                metadata={"source": "mammal-pets-doc"}),
        Document(page_content="Попугаи — умные птицы, которые способны имитировать человеческую речь.", 
                metadata={"source": "bird-pets-doc"})
    ]
    
    print("Создание векторного хранилища Chroma...")
    
    # Создаем векторное хранилище Chroma
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Выполняем поиск по схожести
    print("\nПоиск похожих документов для запроса 'кошка':")
    results = db.similarity_search("кошка", k=2)
    
    print("\nРезультаты поиска:")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content} [Источник: {doc.metadata.get('source')}]")
    
    # Выполняем поиск с оценкой схожести
    print("\nПоиск с оценкой схожести для запроса 'домашний питомец':")
    results_with_scores = db.similarity_search_with_score("домашний питомец", k=4)
    
    print("\nРезультаты поиска с оценкой:")
    for i, (doc, score) in enumerate(results_with_scores):
        print(f"{i+1}. Оценка: {score:.2f} - {doc.page_content} [Источник: {doc.metadata.get('source')}]")
    
    print("\nДемонстрация завершена!")

if __name__ == "__main__":
    demo_gigachat_embeddings() 