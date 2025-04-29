from gigachat_langchain import GigaChatLangchain
from langchain_core.documents import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
import os
import time
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

def main():
    # Инициализируем GigaChatLangchain
    print("Инициализация GigaChatLangchain...")
    gigachat = GigaChatLangchain(verify_ssl=False)
    
    # Получаем модель эмбеддингов и чатовую модель
    embeddings = gigachat.get_embeddings()
    llm = gigachat.get_chat_model()
    
    # Создаем загрузку документов
    documents_dir = "documents"
    documents = load_documents(documents_dir)
    
    # Создаем векторное хранилище
    print("Создание векторного хранилища...")
    db = create_vector_store(documents, embeddings)
    
    # Создаем RAG-цепочку с инструктированием поиска
    print("Создание RAG-цепочки...")
    qa_chain = create_rag_chain(db, llm)
    
    # Демонстрация запросов и получения ответов
    demo_questions = [
        "Требуется информация о способах доставки заказа",
        "Требуется информация о способах оплаты",
        "Требуется информация о контактах службы поддержки"
    ]
    
    print("\n=== Демонстрация работы RAG на основе GigaChat ===\n")
    
    for question in demo_questions:
        print(f"\nВопрос: {question}")
        
        # Задержка, чтобы не превысить лимиты API
        time.sleep(1)
        
        # Получаем ответ от RAG
        start_time = time.time()
        response = qa_chain.invoke({"query": question})
        end_time = time.time()
        
        print(f"\nОтвет: {response['result']}")
        print(f"Время обработки: {end_time - start_time:.2f} секунд")
        print("-" * 50)

def load_documents(documents_dir):
    """Загружает текстовые документы из указанной директории"""
    documents = []
    
    for filename in os.listdir(documents_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(documents_dir, filename)
            
            # Читаем содержимое файла
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Создаем объект документа
            doc = LangchainDocument(
                page_content=text,
                metadata={"source": filename}
            )
            documents.append(doc)
    
    # Разделяем документы на чанки для лучшего поиска
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    
    # Разбиваем документы на чанки
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Загружено {len(documents)} документов, создано {len(chunked_documents)} чанков")
    
    return chunked_documents

def create_vector_store(documents, embeddings):
    """Создает векторное хранилище из документов"""
    # Проверяем наличие директории для Chroma и удаляем ее, если она существует
    if os.path.exists("chroma_rag_db"):
        import shutil
        shutil.rmtree("chroma_rag_db")
        print("Удаление старой базы Chroma")
    
    # Создаем векторное хранилище Chroma
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="chroma_rag_db"
    )
    
    # Сохраняем векторное хранилище
    vector_store.persist()
    print(f"Векторное хранилище создано с {len(documents)} документами")
    
    return vector_store

def create_rag_chain(vector_store, llm):
    """Создает RAG-цепочку для ответов на вопросы"""
    # Шаблон промпта с инструкциями для извлечения информации
    prompt_template = """
    Ты - эксперт по анализу документов. Твоя задача - найти и проанализировать информацию из документов, соответствующую запросу.

    Запрос пользователя: {question}

    Контекст из документов:
    {context}

    Инструкции:
    1. Внимательно проанализируй предоставленный контекст из документов
    2. Найди информацию, которая соответствует запросу пользователя
    3. Если информация найдена, предоставь подробный ответ на основе найденной информации
    4. Изложи найденную информацию в структурированном виде, с деталями и пояснениями
    5. Если информация найдена в нескольких документах, объедини ее в целостный ответ

    ВАЖНО: 
    - Отвечай ТОЛЬКО на основе информации из предоставленных документов
    - Не добавляй информацию из своих знаний, если её нет в документах
    - Твой ответ должен быть полезным и исчерпывающим
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Создаем retriever с настройками
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 4,  # Количество документов для поиска
            "search_type": "similarity"
        }
    )
    
    # Создаем класс InstructionRetriever для улучшения поиска
    class InstructionRetriever:
        def __init__(self, base_retriever):
            self.base_retriever = base_retriever
            
        def get_relevant_documents(self, query):
            # Добавляем инструкцию для retrieval-задачи
            instruction = "Дан вопрос, необходимо найти информацию в документах:\n"
            augmented_query = f"{instruction}{query}"
            return self.base_retriever.get_relevant_documents(augmented_query)
    
    # Создаем retriever с инструкциями
    instructed_retriever = InstructionRetriever(retriever)
    
    # Создаем RetrievalQA цепочку
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Используем "stuff" для объединения всех чанков в один контекст
        retriever=instructed_retriever,
        chain_type_kwargs={"prompt": prompt}
    )

if __name__ == "__main__":
    main() 