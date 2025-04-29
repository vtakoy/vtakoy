"""
Модуль для аутентификации в GigaChat API с улучшенной обработкой ошибок и retry-логикой
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
    """
    Класс для безопасной работы с GigaChat API
    с поддержкой повторных попыток и улучшенным управлением SSL
    """
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        verify_ssl: bool = False,
        timeout: int = 30,
        max_retries: int = 3
    ):
        load_dotenv()
        
        # Инициализация параметров
        self.client_id = client_id or os.getenv("GIGACHAT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("GIGACHAT_CLIENT_SECRET")
        self.auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        self.api_base_url = "https://gigachat.devices.sberbank.ru/api/v1"
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.max_retries = max_retries
        self.auth_token: Optional[str] = None
        self.scope = "GIGACHAT_API_PERS"

        self._validate_credentials()
        self._initialize_session()

    def _validate_credentials(self) -> None:
        """Проверка наличия обязательных учетных данных"""
        if not all([self.client_id, self.client_secret]):
            error_msg = "Необходимо указать GIGACHAT_CLIENT_ID и GIGACHAT_CLIENT_SECRET в .env"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _initialize_session(self) -> None:
        """Инициализация HTTP-сессии с повторными попытками"""
        self.session = requests.Session()
        self.session.verify = False  # Полное отключение проверки SSL
        
        # Настройка адаптера с повторными попытками
        adapter = requests.adapters.HTTPAdapter(
            max_retries=self.max_retries
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=(
            retry_if_exception_type(requests.exceptions.ConnectionError) |
            retry_if_exception_type(requests.exceptions.Timeout)
        ),
        before=before_log(logger, logging.DEBUG)
    )
    def get_token(self) -> str:
        """Получение токена доступа с повторными попытками"""
        try:
            # Формирование Basic Auth
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
            
            token_data = response.json()
            if not (access_token := token_data.get("access_token")):
                raise ValueError("Токен отсутствует в ответе сервера")
            
            self.auth_token = access_token
            logger.info("✅ Токен успешно получен")
            return self.auth_token

        except requests.exceptions.HTTPError as e:
            error_msg = (
                f"HTTP Error {e.response.status_code}: {e.response.reason}\n"
                f"URL: {e.response.url}\n"
                f"Response: {e.response.text[:500]}"
            )
            logger.error(error_msg)
            raise
        except Exception as e:
            logger.error(f"Ошибка получения токена: {str(e)}")
            raise

    def get_headers(self) -> Dict[str, str]:
        """Генерирует заголовки запроса с автоматическим обновлением токена"""
        if not self.auth_token:
            self.get_token()
        return {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(requests.exceptions.HTTPError),
        before=before_log(logger, logging.DEBUG)
    )
    def chat_completion(self, message: str, **kwargs) -> str:
        """Отправка запроса в GigaChat с обработкой 401 ошибки"""
        try:
            url = f"{self.api_base_url}/chat/completions"
            data = {
                "messages": [{"role": "user", "content": message}],
                "model": kwargs.get("model", "GigaChat"),
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1500)
            }

            response = self.session.post(
                url,
                headers=self.get_headers(),
                json=data,
                timeout=self.timeout
            )

            # Обновление токена при 401 ошибке
            if response.status_code == 401:
                logger.warning("Обновление просроченного токена...")
                self.auth_token = None
                return self.chat_completion(message, **kwargs)

            response.raise_for_status()
            
            return response.json()['choices'][0]['message']['content']

        except requests.exceptions.HTTPError as e:
            logger.error(f"Ошибка запроса: {str(e)}")
            if e.response is not None:
                logger.error(f"Ответ сервера: {e.response.text[:500]}")
            raise
        except Exception as e:
            logger.error(f"Общая ошибка: {str(e)}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()