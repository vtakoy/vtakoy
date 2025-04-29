"""
Модуль для аутентификации в GigaChat API
"""

import requests
import os
import ssl
import urllib3
import uuid
import base64
from dotenv import load_dotenv
from typing import Optional

# Отключаем предупреждения SSL для urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Модифицируем настройки SSL для Python
ssl._create_default_https_context = ssl._create_unverified_context

class GigaChatAuth:
    """
    Класс для аутентификации в GigaChat и получения токена доступа
    """
    
    def __init__(self, client_id=None, client_secret=None, disable_ssl_verification=True):
        """
        Инициализация с идентификатором клиента и секретом.
        Если не указаны, будут загружены из переменных окружения.
        
        Args:
            client_id (str, optional): ID клиента GigaChat
            client_secret (str, optional): Секрет клиента GigaChat
            disable_ssl_verification (bool): Отключить проверку SSL сертификатов
        """
        load_dotenv()  # Загружаем переменные окружения
        
        self.client_id = client_id or os.getenv("GIGACHAT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("GIGACHAT_CLIENT_SECRET")
        
        # URL для OAuth авторизации
        self.auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        
        # URL для API
        self.api_base_url = "https://gigachat.devices.sberbank.ru/api/v1"
        
        self.verify_ssl = not disable_ssl_verification
        self.auth_token = None
        self.scope = "GIGACHAT_API_PERS"  # Доступ для физических лиц
        
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Необходимо указать GIGACHAT_CLIENT_ID и GIGACHAT_CLIENT_SECRET в .env файле или передать в конструктор"
            )
        
        # Получаем токен при создании экземпляра
        try:
            self.get_token()
        except ConnectionError as e:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Ошибка авторизации: {str(e)}")
            print("Убедитесь, что в файле .env указаны правильные GIGACHAT_CLIENT_ID и GIGACHAT_CLIENT_SECRET")
            print("Также проверьте доступность API GigaChat (может требоваться VPN)")
    
    def get_token(self):
        """
        Получение токена авторизации от GigaChat API
        
        Returns:
            str: Токен авторизации
        """
        # Создаем Basic Authorization заголовок
        auth_str = f"{self.client_id}:{self.client_secret}"
        auth_bytes = auth_str.encode('ascii')
        base64_bytes = base64.b64encode(auth_bytes)
        base64_auth = base64_bytes.decode('ascii')
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {base64_auth}"
        }
        
        payload = {
            "scope": self.scope,
        }
        
        try:
            print(f"Выполняется запрос для получения токена авторизации GigaChat...")
            
            response = requests.post(
                self.auth_url, 
                headers=headers, 
                data=payload, 
                verify=self.verify_ssl,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"Статус ответа: {response.status_code}")
                print(f"Тело ответа: {response.text}")
            
            response.raise_for_status()
            response_data = response.json()
            
            self.auth_token = response_data.get("access_token")
            
            if not self.auth_token:
                raise ValueError("Токен не был получен из ответа API")
                
            print("Токен авторизации успешно получен")
            return self.auth_token
            
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при получении токена: {str(e)}")
            print("Данные запроса:")
            print(f"- URL: {self.auth_url}")
            print(f"- Authorization: Basic {base64_auth[:10]}...")
            raise ConnectionError(f"Ошибка при получении токена авторизации: {str(e)}")
    
    def get_headers(self):
        """
        Получение заголовков для запросов к GigaChat API
        
        Returns:
            dict: Заголовки с токеном авторизации
        """
        if not self.auth_token:
            self.get_token()
            
        return {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def get_models(self) -> Optional[dict]:
        """Get available GigaChat models."""
        url = f"{self.api_base_url}/models"
        headers = {
            'Authorization': f'Bearer {self.get_token()}',
            'Accept': 'application/json'
        }

        try:
            response = requests.get(url, headers=headers, verify=self.verify_ssl, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Ошибка при получении списка моделей: {str(e)}")
            return None

    def get_balance(self) -> Optional[dict]:
        """Get token balance."""
        url = f"{self.api_base_url}/tokens/balance"
        headers = {
            'Authorization': f'Bearer {self.get_token()}',
            'Accept': 'application/json'
        }

        try:
            response = requests.get(url, headers=headers, verify=self.verify_ssl, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Ошибка при получении баланса: {str(e)}")
            return None

    def chat_completion(self, message: str) -> str:
        """Send message to GigaChat API and get response."""
        try:
            access_token = self.get_token()
            url = f"{self.api_base_url}/chat/completions"
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {access_token}',
                'RqUID': str(uuid.uuid4())
            }
            
            data = {
                'messages': [
                    {
                        'role': 'user',
                        'content': message
                    }
                ],
                'model': 'GigaChat',
                'temperature': 0.7,
                'max_tokens': 1500
            }
            
            response = requests.post(
                url,
                headers=headers,
                json=data,
                verify=self.verify_ssl,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            elif response.status_code == 401:
                # Если токен истек, пробуем получить новый
                self.auth_token = None
                return self.chat_completion(message)
            else:
                raise Exception(f"Ошибка при отправке сообщения: {response.status_code} {response.text}")
                
        except Exception as e:
            print(f"Ошибка при отправке сообщения: {str(e)}")
            raise