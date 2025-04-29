import os
from dotenv import load_dotenv
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

def demo_gigachat_embeddings_with_instructions():
    """
    Демонстрация работы с GigaChatEmbeddings с использованием инструкций
    для улучшения качества retrieval-задач согласно документации GigaChat
    """
    print("Демонстрация работы с GigaChatEmbeddings с инструкциями")
    
    # Получаем учетные данные GigaChat
    client_id = os.getenv("GIGACHAT_CLIENT_ID")
    client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("Ошибка: Не указаны GIGACHAT_CLIENT_ID или GIGACHAT_CLIENT_SECRET в .env файле")
        return
    
    print("Инициализация GigaChatEmbeddings...")
    
    # Создаем экземпляр GigaChatEmbeddings, используем модель EmbeddingsGigaR вместо Embeddings
    credentials = f"{client_id}:{client_secret}"
    embeddings = GigaChatEmbeddings(
        credentials=credentials,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS",
        model="EmbeddingsGigaR"  # Используем улучшенную модель для эмбеддингов
    )
    
    # Создаем тестовые документы
    documents = [
        Document(page_content="Москва - столица России, крупнейший по численности населения город страны", 
                metadata={"source": "cities-doc"}),
        Document(page_content="Санкт-Петербург - второй по численности населения город России, важный культурный центр", 
                metadata={"source": "cities-doc"}),
        Document(page_content="Борщ - традиционный славянский суп, популярный в национальных кухнях многих стран", 
                metadata={"source": "food-doc"}),
        Document(page_content="Пельмени - блюдо в виде отварных изделий из теста с начинкой из рубленого мяса", 
                metadata={"source": "food-doc"})
    ]
    
    print("Создание векторного хранилища Chroma...")
    
    # Создаем векторное хранилище Chroma
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db_instructions"
    )
    
    # Примеры запросов с инструкциями для разных типов задач
    
    # 1. Запрос без инструкции
    print("\n1. Базовый запрос без инструкции:")
    query = "еда в России"
    print(f"Запрос: {query}")
    
    results = db.similarity_search(query, k=2)
    print("Результаты:")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content} [Источник: {doc.metadata.get('source')}]")
    
    # 2. Запрос с инструкцией для retrieval-задачи
    print("\n2. Запрос с инструкцией для retrieval-задачи:")
    query_with_instruction = "Дан вопрос, необходимо найти абзац текста с ответом \nвопрос: какие блюда популярны в России?"
    print(f"Запрос с инструкцией: {query_with_instruction}")
    
    results = db.similarity_search(query_with_instruction, k=2)
    print("Результаты:")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content} [Источник: {doc.metadata.get('source')}]")
    
    # 3. Запрос с инструкцией для симметричной задачи
    print("\n3. Запрос с инструкцией для симметричной задачи:")
    query_with_symmetric_instruction = "Классифицируй текст по теме: еда или города \nтекст: города России"
    print(f"Запрос с инструкцией: {query_with_symmetric_instruction}")
    
    results = db.similarity_search(query_with_symmetric_instruction, k=2)
    print("Результаты:")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content} [Источник: {doc.metadata.get('source')}]")
    
    print("\nДемонстрация завершена!")

if __name__ == "__main__":
    demo_gigachat_embeddings_with_instructions() 