"""
Модуль для аутентификации в GigaChat API (внутренний контур Сбера)
"""

import requests
import os
import ssl
import urllib3
import uuid
import base64
from dotenv import load_dotenv
from typing import Optional

# Отключаем предупреждения SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

class GigaChatAuth:
    """
    Класс для работы с GigaChat API
    """
    
    def __init__(self, client_id=None, client_secret=None, disable_ssl_verification=True):
        load_dotenv()
        
        self.client_id = client_id or os.getenv("GIGACHAT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("GIGACHAT_CLIENT_SECRET")
        self.auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        self.api_base_url = "https://gigachat.devices.sberbank.ru/api/v1"
        self.verify_ssl = not disable_ssl_verification
        self.auth_token = None
        self.scope = "GIGACHAT_API_PERS"

        if not self.client_id or not self.client_secret:
            raise ValueError("Необходимо указать учетные данные в .env")

        try:
            self.get_token()
        except Exception as e:
            print(f"❌ Ошибка инициализации: {str(e)}")
            self.auth_token = None

    def get_token(self):
        """Получение токена доступа"""
        try:
            auth_str = f"{self.client_id}:{self.client_secret}"
            base64_auth = base64.b64encode(auth_str.encode()).decode()
            
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "RqUID": str(uuid.uuid4()),
                "Authorization": f"Basic {base64_auth}"
            }

            response = requests.post(
                self.auth_url,
                headers=headers,
                data={"scope": self.scope},
                verify=self.verify_ssl,
                timeout=15
            )

            response.raise_for_status()
            self.auth_token = response.json().get("access_token")
            
            if not self.auth_token:
                raise ValueError("Токен не получен в ответе")
                
            print("✅ Токен успешно получен")
            return self.auth_token

        except requests.exceptions.HTTPError as e:
            print(f"⚠️ Ошибка HTTP: {e.response.status_code}")
            print(f"Ответ сервера: {e.response.text[:300]}...")
            raise
        except Exception as e:
            print(f"🚨 Ошибка подключения: {str(e)}")
            raise

    def get_headers(self):
        """Генерирует заголовки запроса"""
        if not self.auth_token:
            self.get_token()
        return {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def chat_completion(self, message: str) -> str:
        """Отправка запроса в GigaChat"""
        try:
            headers = self.get_headers()
            data = {
                "messages": [{"role": "user", "content": message}],
                "model": "GigaChat",
                "temperature": 0.7,
                "max_tokens": 1500
            }
            
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                json=data,
                verify=self.verify_ssl,
                timeout=30
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                self.auth_token = None
                return self.chat_completion(message)
            raise
        except Exception as e:
            print(f"Ошибка запроса: {str(e)}")
            raise

    # Другие методы остаются аналогичными

    def get_models(self) -> Optional[dict]:
        """Получение списка доступных моделей"""
        try:
            response = requests.get(
                f"{self.api_base_url}/models",
                headers=self.get_headers(),
                verify=self.verify_ssl,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Ошибка получения моделей: {str(e)}")
            return None

    def get_balance(self) -> Optional[dict]:
        """Проверка баланса токенов"""
        try:
            response = requests.get(
                f"{self.api_base_url}/tokens/balance",
                headers=self.get_headers(),
                verify=self.verify_ssl,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Ошибка проверки баланса: {str(e)}")
            return None