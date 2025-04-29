from langchain_gigachat.embeddings import GigaChatEmbeddings
from langchain_gigachat.chat_models import GigaChat
from typing import List, Dict, Any, Optional
import os
import base64
from dotenv import load_dotenv

# Загружаем переменные окружения, если еще не загружены
load_dotenv()

class GigaChatLangchain:
    """Класс для работы с GigaChat через langchain-gigachat"""

    def __init__(
        self,
        credentials=None,
        client_id=None,
        client_secret=None,
        verify_ssl=False, 
        scope="GIGACHAT_API_PERS"
    ):
        """
        Инициализация класса
        
        Args:
            credentials: учетные данные для GigaChat в формате base64 или строка 'client_id:client_secret'
            client_id: ID клиента для GigaChat (альтернатива credentials)
            client_secret: Секрет клиента для GigaChat (альтернатива credentials)
            verify_ssl: проверка SSL сертификата
            scope: область доступа
        """
        # Создаем credentials из client_id и client_secret, если они предоставлены
        if not credentials and client_id and client_secret:
            # Создаем строку и кодируем в base64
            auth_str = f"{client_id}:{client_secret}"
            auth_bytes = auth_str.encode('ascii')
            credentials = base64.b64encode(auth_bytes).decode('ascii')
            
        self.credentials = credentials
        self.verify_ssl = verify_ssl
        self.scope = scope

    def get_embeddings(self):
        """Возвращает экземпляр GigaChatEmbeddings для векторизации текстов"""
        return GigaChatEmbeddings(
            credentials=self.credentials,
            verify_ssl_certs=self.verify_ssl
        )
    
    def get_chat_model(self) -> GigaChat:
        """
        Создает и возвращает модель чата GigaChat.
        
        Returns:
            Объект GigaChat для генерации ответов
        """
        return GigaChat(
            credentials=self.credentials,
            verify_ssl_certs=self.verify_ssl,
            scope=self.scope
        )
    
    def embed_text(self, text: str) -> List[float]:
        """
        Создает векторное представление текста.
        
        Args:
            text: Текст для векторизации
            
        Returns:
            Список чисел (вектор) для переданного текста
        """
        embeddings = self.get_embeddings()
        result = embeddings.embed_query(text)
        return result
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Создает векторные представления для списка текстов.
        
        Args:
            texts: Список текстов для векторизации
            
        Returns:
            Список векторов для переданных текстов
        """
        embeddings = self.get_embeddings()
        result = embeddings.embed_documents(texts)
        return result
    
    def chat_completion(self, message: str) -> str:
        """
        Отправляет сообщение GigaChat и получает ответ.
        
        Args:
            message: Текст сообщения
            
        Returns:
            Ответ от модели
        """
        model = self.get_chat_model()
        response = model.invoke(message)
        return response.content 