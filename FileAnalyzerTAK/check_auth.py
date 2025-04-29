"""
Скрипт для проверки авторизации в GigaChat API
"""

import requests
import os
import ssl
import urllib3
import uuid
import json
import base64
from dotenv import load_dotenv

# Отключаем предупреждения SSL для urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Модифицируем настройки SSL для Python
ssl._create_default_https_context = ssl._create_unverified_context

def check_auth():
    """Проверяет авторизацию в GigaChat API"""
    print("\n=== Проверка авторизации в GigaChat API ===\n")
    
    # Загружаем переменные окружения
    load_dotenv()
    
    # Получаем учетные данные из .env
    client_id = os.getenv("GIGACHAT_CLIENT_ID")
    client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("ОШИБКА: Не найдены переменные GIGACHAT_CLIENT_ID и/или GIGACHAT_CLIENT_SECRET в .env")
        print("\nУбедитесь, что файл .env создан и содержит следующие строки:")
        print("GIGACHAT_CLIENT_ID=ваш_client_id")
        print("GIGACHAT_CLIENT_SECRET=ваш_client_secret")
        return
    
    print(f"Client ID найден: {client_id[:5]}..." if len(client_id) > 5 else f"Client ID найден: {client_id}")
    print(f"Client Secret найден: {client_secret[:5]}..." if len(client_secret) > 5 else f"Client Secret найден: {client_secret}")
    
    # URL для OAuth авторизации
    auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    
    # Создаем Basic Authorization заголовок
    auth_str = f"{client_id}:{client_secret}"
    auth_bytes = auth_str.encode('ascii')
    base64_bytes = base64.b64encode(auth_bytes)
    base64_auth = base64_bytes.decode('ascii')
    
    # Формируем заголовки
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
        "Authorization": f"Basic {base64_auth}"
    }
    
    # Формируем данные для запроса
    payload = {
        "scope": "GIGACHAT_API_PERS",
    }
    
    print("\nОтправка запроса для получения токена...")
    print(f"URL: {auth_url}")
    print(f"Authorization: Basic {base64_auth[:10]}...")
    
    try:
        # Отправляем запрос
        response = requests.post(
            auth_url,
            headers=headers,
            data=payload,
            verify=False,
            timeout=30
        )
        
        # Выводим информацию о статусе запроса
        print(f"Статус ответа: {response.status_code}")
        
        if response.status_code == 200:
            print("Авторизация успешна!")
            response_data = response.json()
            token = response_data.get("access_token")
            if token:
                print(f"Получен токен: {token[:10]}...")
                
                # Проверяем API методы
                check_api_methods(token)
            else:
                print("ОШИБКА: Токен не получен в ответе API")
                print(f"Текст ответа: {response.text}")
        else:
            print("ОШИБКА: Авторизация не удалась")
            print(f"Текст ответа: {response.text}")
    
    except Exception as e:
        print(f"ОШИБКА: Не удалось подключиться к API: {str(e)}")
        print("\nВозможные причины:")
        print("1. Неверные учетные данные в .env")
        print("2. Проблемы с сетевым подключением")
        print("3. Необходим VPN для доступа к API")
        print("4. Изменился URL API")

def check_api_methods(token):
    """Проверяет доступные методы API GigaChat"""
    print("\n=== Проверка доступа к API методам ===\n")
    
    # URL для списка моделей
    models_url = "https://gigachat.devices.sberbank.ru/api/v1/models"
    
    # Формируем заголовки
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        # Проверяем доступ к списку моделей
        print("Проверка доступа к списку моделей...")
        response = requests.get(
            models_url,
            headers=headers,
            verify=False,
            timeout=30
        )
        
        if response.status_code == 200:
            print("Доступ к API успешен!")
            models = response.json()
            print(f"Доступные модели: {json.dumps(models, indent=2, ensure_ascii=False)}")
        else:
            print(f"ОШИБКА: Не удалось получить список моделей. Статус: {response.status_code}")
            print(f"Текст ответа: {response.text}")
    
    except Exception as e:
        print(f"ОШИБКА при доступе к API: {str(e)}")

if __name__ == "__main__":
    check_auth() 