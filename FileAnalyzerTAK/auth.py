import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

import uuid
import base64
import logging
from typing import Optional, Dict
from dotenv import load_dotenv
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import urllib3

# Отключение предупреждений SSL
urllib3.disable_warnings()

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
        
        # Определяем режим работы
        work_location = os.getenv("WORK_LOCATION", "home").lower()
        self.logger = logging.getLogger('GigaChatAuth')
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Режим работы: {'на работе' if work_location == 'work' else 'дома'}")
        
        # Устанавливаем URL в зависимости от режима работы
        if work_location == "work":
            self.auth_url = os.getenv("GIGACHAT_AUTH_URL", "https://sm-auth-sd.prom-88-89-apps.ocp-geo.ocp.sigma.sbrf.ru/api/v2/oauth")
            self.api_url = os.getenv("GIGACHAT_BASE_URL", "https://gigachat.devices.sberbank.ru/api/v1")
        else:
            self.auth_url = os.getenv("GIGACHAT_AUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
            self.api_url = os.getenv("GIGACHAT_BASE_URL", "https://gigachat.devices.sberbank.ru/api/v1")
        
        self.logger.info(f"Используется URL аутентификации: {self.auth_url}")
        self.logger.info(f"Используется URL API: {self.api_url}")
        
        self.timeout = timeout
        self.max_retries = max_retries
        self.auth_token = None
        self.scope = "GIGACHAT_API_PERS"
        
        self._validate_credentials()
        self._init_session()

    def _validate_credentials(self):
        if not all([self.client_id, self.client_secret]):
            raise ValueError("Требуются client_id и client_secret")

    def _init_session(self):
        self.session = requests.Session()
        self.session.verify = False
        adapter = requests.adapters.HTTPAdapter(max_retries=self.max_retries)
        self.session.mount("https://", adapter)

    def _generate_auth_key(self) -> str:
        """Generate authorization key for GigaChat API."""
        auth_string = f"{self.client_id}:{self.client_secret}"
        return base64.b64encode(auth_string.encode()).decode()

    def _get_auth_headers(self) -> dict:
        """Get headers for authentication request."""
        return {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'Authorization': f'Basic {self._generate_auth_key()}',
            'RqUID': str(uuid.uuid4())
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException)
    )
    def get_token(self) -> str:
        """Получение токена аутентификации"""
        try:
            data = {
                'scope': self.scope,
                'grant_type': 'client_credentials'  # Ключевое изменение
            }

            response = self.session.post(
                self.auth_url,
                headers=self._get_auth_headers(),
                data=data,
                timeout=self.timeout
            )

            response.raise_for_status()
            self.auth_token = response.json().get("access_token")
            if not self.auth_token:
                raise ValueError("Токен не получен")
            
            self.logger.info("Токен успешно получен")
            return self.auth_token

        except requests.exceptions.HTTPError as e:
            error_msg = f"Ошибка получения токена: {e.response.status_code}"
            if e.response.text:
                error_msg += f" - {e.response.text[:200]}"
            self.logger.error(error_msg)
            raise
        except Exception as e:
            self.logger.error(f"Ошибка получения токена: {str(e)}")
            raise

    def get_headers(self) -> Dict[str, str]:
        """Получение заголовков для запросов"""
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
        retry=retry_if_exception_type(requests.exceptions.HTTPError)
    )
    def chat_completion(self, message: str, **kwargs) -> str:
        """Запрос к чат-API"""
        try:
            response = self.session.post(
                f"{self.api_url}/chat/completions",
                headers=self.get_headers(),
                json={
                    "messages": [{"role": "user", "content": message}],
                    "model": kwargs.get("model", "GigaChat:latest"),  # Изменено на :latest
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
            self.logger.error(f"Ошибка запроса: {str(e)}")
            raise

    def get_models(self) -> Optional[dict]:
        """Получение списка доступных моделей"""
        try:
            response = self.session.get(
                f"{self.api_url}/models",
                headers=self.get_headers(),
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Ошибка получения моделей: {str(e)}")
            return None

    def get_balance(self) -> Optional[dict]:
        """Проверка баланса токенов"""
        try:
            response = self.session.get(
                f"{self.api_url}/tokens/balance",
                headers=self.get_headers(),
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Ошибка проверки баланса: {str(e)}")
            return None