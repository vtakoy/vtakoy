"""
Модуль для аутентификации в GigaChat API
"""

import os
import uuid
import base64
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_log
)
from urllib3.exceptions import InsecureRequestWarning
import urllib3

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Отключаем предупреждения SSL
urllib3.disable_warnings(InsecureRequestWarning)

class GigaChatAuth:
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        verify_ssl: bool = False,
        timeout: int = 30,
        max_retries: int = 3
    ):
        load_dotenv()
        
        self.client_id = client_id or os.getenv("GIGACHAT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("GIGACHAT_CLIENT_SECRET")
        self.auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        self.api_base_url = "https://gigachat.devices.sberbank.ru/api/v1"
        self.timeout = timeout
        self.max_retries = max_retries
        self.auth_token: Optional[str] = None
        self.scope = "GIGACHAT_API_PERS"

        self._validate_credentials()
        self._initialize_session()

    def _validate_credentials(self) -> None:
        if not all([self.client_id, self.client_secret]):
            raise ValueError("Необходимо указать GIGACHAT_CLIENT_ID и GIGACHAT_CLIENT_SECRET")

    def _initialize_session(self) -> None:
        self.session = requests.Session()
        self.session.verify = False
        
        adapter = requests.adapters.HTTPAdapter(
            max_retries=self.max_retries
        )
        self.session.mount("https://", adapter)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=(
            retry_if_exception_type(requests.exceptions.ConnectionError) |
            retry_if_exception_type(requests.exceptions.Timeout)
        )
    )
    def get_token(self) -> str:
        try:
            credentials = f"{self.client_id}:{self.client_secret}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "RqUID": str(uuid.uuid4()),
                "Authorization": f"Basic {encoded_credentials}"
            }

            response = self.session.post(
                self.auth_url,
                headers=headers,
                data={"scope": self.scope},
                timeout=self.timeout
            )

            response.raise_for_status()
            self.auth_token = response.json().get("access_token")
            if not self.auth_token:
                raise ValueError("Токен отсутствует в ответе")
            
            logger.info("Токен успешно получен")
            return self.auth_token

        except Exception as e:
            logger.error(f"Ошибка получения токена: {str(e)}")
            raise

    def get_headers(self) -> Dict[str, str]:
        if not self.auth_token:
            self.get_token()
        return {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json"
        }

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(requests.exceptions.HTTPError)
    )
    def chat_completion(self, message: str, **kwargs) -> str:
        try:
            response = self.session.post(
                f"{self.api_base_url}/chat/completions",
                headers=self.get_headers(),
                json={
                    "messages": [{"role": "user", "content": message}],
                    "model": kwargs.get("model", "GigaChat"),
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 1500)
                },
                timeout=self.timeout
            )

            if response.status_code == 401:
                self.auth_token = None
                return self.chat_completion(message, **kwargs)

            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']

        except Exception as e:
            logger.error(f"Ошибка запроса: {str(e)}")
            raise