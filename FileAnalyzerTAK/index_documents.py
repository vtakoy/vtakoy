"""
index_documents.py — отдельный скрипт для индексации документов с использованием нового API ChromaDB
"""
import os
import logging
from dotenv import load_dotenv
from auth import GigaChatAuth
from document_analyzer import DocumentAnalyzer
from chromadb import PersistentClient
from langchain_community.vectorstores import Chroma

# Конфигурация
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "documents_collection"

if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    
    # Инициализация клиента GigaChat
    client_id = os.getenv("GIGACHAT_CLIENT_ID")
    client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")
    client = GigaChatAuth(client_id, client_secret, verify_ssl=False)

    # Инициализация DocumentAnalyzer
    analyzer = DocumentAnalyzer(client)

    # Чтение и индексация документов
    try:
        logging.info("Запуск индексации документов через отдельный скрипт...")
        analyzer.read_documents()
        logging.info("Индексация завершена!")
    except Exception as e:
        logging.error(f"Ошибка при индексации документов: {str(e)}")
        exit(1)
