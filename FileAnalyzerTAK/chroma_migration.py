import os
import shutil
from dotenv import load_dotenv
from auth import GigaChatAuth
from gigachat_langchain import GigaChatLangchain
from file import DocumentAnalyzer
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

class ChromaDocumentAnalyzer(DocumentAnalyzer):
    """Модифицированный DocumentAnalyzer, использующий Chroma вместо FAISS"""
    
    def _create_vector_store(self, documents):
        """Создает векторное хранилище Chroma из документов"""
        try:
            self.logger.log_operation("Создание векторного хранилища Chroma", f"Обработка {len(documents)} документов")
            
            # Удаляем существующую базу Chroma, если она есть
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
                self.logger.log_operation("Удаление старого хранилища Chroma", "Успешно")
            
            # Создаем векторное хранилище Chroma
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
            
            # Сохраняем векторное хранилище
            vector_store.persist()
            
            self.logger.log_operation("Векторное хранилище Chroma создано", "Успешно")
            return vector_store
            
        except Exception as e:
            self.logger.log_error("Ошибка при создании векторного хранилища Chroma", str(e))
            raise
    
    def _create_qa_chain(self, vector_store):
        """Создает цепочку для ответов на вопросы с использованием Chroma"""
        # Используем тот же код, что и в родительском классе, но с vector_store типа Chroma
        prompt_template = """
        Ты - эксперт по анализу документов. Твоя задача - найти и проанализировать информацию из документов, соответствующую запросу.

        Запрос пользователя: {question}

        Контекст из документов:
        {context}

        Инструкции:
        1. Внимательно проанализируй предоставленный контекст из документов
        2. Найди информацию, которая соответствует запросу пользователя
        3. ВАЖНО: ДАЖЕ ЕСЛИ ИНФОРМАЦИЯ ПРЕДОСТАВЛЕНА ЧАСТИЧНО ИЛИ КОСВЕННО, ИСПОЛЬЗУЙ ЕЁ ДЛЯ ОТВЕТА
        4. Если информация найдена, укажи:
           - Название документа, где найдена информация
           - Точную страницу (если доступна) и раздел, где найдена информация
           - Подробный ответ на основе найденной информации
        5. Изложи найденную информацию в структурированном виде, с деталями и пояснениями
        6. Если информация найдена в нескольких документах, укажи все источники

        Формат ответа:
        "Информация найдена на странице <номер страницы> в разделе <Наименование раздела> документа <Имя файла>: <Ответ с найденной информацией>"

        ВАЖНО: 
        - Отвечай ТОЛЬКО на основе информации из предоставленных документов
        - ИСПОЛЬЗУЙ ЛЮБУЮ РЕЛЕВАНТНУЮ ИНФОРМАЦИЮ ИЗ КОНТЕКСТА, даже если она не точно соответствует запросу
        - Можешь перефразировать и интерпретировать информацию для лучшего понимания
        - Не добавляй информацию из своих знаний, если её нет в документах
        - Обязательно укажи источник информации (название документа)
        - Не используй никаких вводных фраз, начинай сразу с ответа
        - НИКОГДА не отвечай "Информация не найдена", если в контексте есть хоть какая-то релевантная информация!
        """
        
        from langchain.prompts import PromptTemplate
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        from langchain.chains import RetrievalQA
        return RetrievalQA.from_chain_type(
            llm=self.client,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 15}),
            chain_type_kwargs={"prompt": prompt}
        )

def migrate_to_chroma():
    """Миграция системы на использование GigaChatEmbeddings с Chroma"""
    print("Начало миграции на Chroma с GigaChatEmbeddings")
    
    # Получаем учетные данные GigaChat
    client_id = os.getenv("GIGACHAT_CLIENT_ID")
    client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("Ошибка: Не указаны GIGACHAT_CLIENT_ID или GIGACHAT_CLIENT_SECRET в .env файле")
        return False
    
    print("Инициализация GigaChatAuth...")
    client = GigaChatAuth(client_id, client_secret, disable_ssl_verification=True)
    
    print("Инициализация ChromaDocumentAnalyzer...")
    analyzer = ChromaDocumentAnalyzer(client)
    
    print("Чтение документов...")
    analyzer.read_documents()
    
    print("Миграция завершена успешно!")
    
    return analyzer

if __name__ == "__main__":
    analyzer = migrate_to_chroma()
    
    if analyzer:
        print("\nТестирование системы:")
        query = "Требуется информация о возможностях системы"
        
        print(f"\nЗапрос: {query}")
        print("\nОтвет:")
        
        try:
            response = analyzer.analyze_documents(query)
            print(response)
        except Exception as e:
            print(f"Ошибка при выполнении запроса: {str(e)}")
    else:
        print("Миграция не выполнена из-за ошибок") 