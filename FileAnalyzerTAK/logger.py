import logging
from datetime import datetime
import os

# Создаем директорию для логов, если её нет
if not os.path.exists('logs'):
    os.makedirs('logs')

# Настраиваем логгер
logger = logging.getLogger('assistant')
logger.setLevel(logging.DEBUG)

# Создаем форматтер для логов
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Хендлер для файла
log_file = f'logs/assistant_{datetime.now().strftime("%Y%m%d")}.log'
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def log_api_request(api_name: str, url: str, params: dict, headers: dict = None):
    """Логирует запрос к API"""
    # Маскируем API ключи в параметрах и заголовках
    safe_params = {k: v[:8] + '...' + v[-8:] if k.lower().endswith('key') else v 
                  for k, v in params.items()}
    
    safe_headers = {}
    if headers:
        safe_headers = {k: v[:8] + '...' + v[-8:] if k.lower() == 'authorization' else v 
                       for k, v in headers.items()}
    
    logger.info(f"\n{api_name} API Request:")
    logger.info(f"URL: {url}")
    logger.info(f"Parameters: {safe_params}")
    if headers:
        logger.info(f"Headers: {safe_headers}")

def log_api_response(api_name: str, status_code: int, response_data: dict = None, error: str = None):
    """Логирует ответ от API"""
    logger.info(f"\n{api_name} API Response:")
    logger.info(f"Status Code: {status_code}")
    if error:
        logger.error(f"Error: {error}")
    elif response_data:
        logger.info(f"Response Data: {str(response_data)[:500]}...") 