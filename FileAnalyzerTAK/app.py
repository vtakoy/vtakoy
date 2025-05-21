import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'

os.environ['CHROMA_TELEMETRY_ENABLED'] = 'False'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

import warnings
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import shutil
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from auth import GigaChatAuth
from document_analyzer import DocumentAnalyzer
import threading
import logging
from datetime import datetime

logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('chromadb.telemetry.posthog').setLevel(logging.CRITICAL)
logging.getLogger('backoff').setLevel(logging.CRITICAL)
logging.getLogger('httpx').setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"app_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def clean_chroma_dir():
    """Очистка папки chroma перед запуском"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chroma_path = os.path.join(current_dir, "chroma")
    try:
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
            logger.info(f"Папка {chroma_path} успешно удалена")
    except Exception as e:
        logger.error(f"Ошибка при очистке папки chroma: {str(e)}")

load_dotenv()

try:
    logger.info("Инициализация GigaChat клиента...")
    client_id = os.getenv("GIGACHAT_CLIENT_ID")
    client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")
    
    # Определяем режим работы
    work_location = os.getenv("WORK_LOCATION", "home").lower()
    logger.info(f"Режим работы: {'на работе' if work_location == 'work' else 'дома'}")
    
    # Устанавливаем URL в зависимости от режима работы
    if work_location == "work":
        auth_url = os.getenv("GIGACHAT_AUTH_URL", "https://sm-auth-sd.prom-88-89-apps.ocp-geo.ocp.sigma.sbrf.ru/api/v2/oauth")
        base_url = os.getenv("GIGACHAT_BASE_URL", "https://gigachat.devices.sberbank.ru/api/v1")
    else:
        auth_url = os.getenv("GIGACHAT_AUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
        base_url = os.getenv("GIGACHAT_BASE_URL", "https://gigachat.devices.sberbank.ru/api/v1")
    
    logger.info(f"Используется URL аутентификации: {auth_url}")
    logger.info(f"Используется URL API: {base_url}")
    
    os.environ.update({
        'GIGACHAT_AUTH_URL': auth_url,
        'GIGACHAT_BASE_URL': base_url
    })
    
    client = GigaChatAuth(client_id, client_secret, verify_ssl=False)
    logger.info("GigaChat клиент успешно инициализирован")
except Exception as e:
    logger.error(f"Ошибка инициализации GigaChat: {str(e)}")
    raise

clean_chroma_dir()

try:
    logger.info("Инициализация DocumentAnalyzer...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    documents_path = os.path.join(current_dir, "documents")
    analyzer = DocumentAnalyzer(client, documents_dir=documents_path)
    
    logger.info("Запуск индексации документов...")
    analyzer.read_documents()
    
    chroma_path = os.path.join(current_dir, "chroma")
    if os.path.exists(chroma_path):
        db_files = set()
        for root, dirs, files in os.walk(chroma_path):
            db_files.update(files)
        
        if 'chroma.sqlite3' in db_files:
            logger.info(f"База данных успешно создана в {chroma_path}")
            if hasattr(analyzer, 'vector_store') and analyzer.vector_store is not None:
                collection = analyzer.vector_store._collection
                if collection:
                    logger.info(f"Векторная база содержит {collection.count()} документов")
        else:
            logger.error(f"Основной файл базы не найден в {chroma_path}")
    else:
        logger.error(f"Папка базы данных не создана: {chroma_path}")
    
    logger.info("Индексация завершена")
except Exception as e:
    logger.error(f"Ошибка инициализации анализатора: {str(e)}")
    analyzer = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    query = request.form.get('query', '')
    
    if not query:
        logger.warning("Получен пустой запрос")
        return jsonify({"status": "error", "message": "Пустой запрос"}), 400
    
    logger.info(f"Получен запрос: {query}")
    
    has_prefix = query.lower().startswith('требуется информация')
    
    if not has_prefix:
        try:
            logger.info("Запрос без префикса, перенаправляю напрямую в GigaChat")
            response = client.chat_completion(query)
            return jsonify({
                "status": "success",
                "results": [{"content": response, "section": "Ответ GigaChat"}]
            })
        except Exception as e:
            logger.error(f"Ошибка запроса к GigaChat: {str(e)}")
            return jsonify({"status": "error", "message": f"Ошибка GigaChat: {str(e)}"}), 500
    
    try:
        logger.info("Запуск RAG анализа...")
        max_execution_time = 30
        result = analyze_documents(query, max_execution_time)
        
        if result["status"] == "timeout":
            return jsonify(result), 408
        
        if result["status"] == "error":
            return jsonify(result), 500
        
        return jsonify(parse_response(result["results"]))
        
    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {str(e)}")
        return jsonify({"status": "error", "message": f"Ошибка обработки: {str(e)}"}), 500

def analyze_documents(query, max_execution_time=30):
    timeout_result = {"status": "timeout", "message": "Превышено время обработки"}
    result = None
    
    def process_query():
        nonlocal result
        try:
            global analyzer
            if not analyzer:
                result = {"status": "error", "message": "Анализатор не инициализирован"}
                return
            
            response = analyzer.analyze_documents(query)
            if isinstance(response, dict) and "error" in response:
                result = {"status": "error", "message": response["error"]}
            else:
                result = {"status": "success", "results": response}
        except Exception as e:
            result = {"status": "error", "message": str(e)}
    
    thread = threading.Thread(target=process_query)
    thread.daemon = True
    thread.start()
    thread.join(timeout=max_execution_time)
    
    if thread.is_alive():
        logger.warning(f"Таймаут ({max_execution_time} сек)")
        return timeout_result
    
    return result if result else {"status": "error", "message": "Неизвестная ошибка"}

def parse_response(response):
    if not response:
        return {"status": "error", "message": "Пустой ответ"}
    
    # Если response - это словарь с ошибкой
    if isinstance(response, dict) and "error" in response:
        return {"status": "error", "message": response["error"]}
    
    # Если response - это словарь с ответом
    if isinstance(response, dict):
        if "answer" in response:
            answer = response["answer"]
            if isinstance(answer, dict):
                # Извлекаем основной текст ответа
                if "result" in answer:
                    response_text = answer["result"]
                else:
                    return {"status": "error", "message": "Неизвестный формат ответа RAG chain"}
                
                # Форматируем источники
                sources = []
                if "source_details" in answer:
                    for source in answer["source_details"]:
                        source_info = []
                        if source.get("source"):
                            source_info.append(f"Документ: {source['source']}")
                        if source.get("section"):
                            source_info.append(f"Раздел: {source['section']}")
                        if source.get("page"):
                            source_info.append(f"Страница: {source['page']}")
                        if source.get("sheet"):
                            source_info.append(f"Лист: {source['sheet']}")
                        if source_info:
                            sources.append(" | ".join(source_info))
                
                # Проверяем наличие фраз о ненайденной информации
                not_found_phrases = [
                    "не найдена", "не найдено", "отсутствует", 
                    "не могу найти", "не удалось найти", 
                    "информация отсутствует", "данные не найдены"
                ]
                response_lower = response_text.lower()
                
                for phrase in not_found_phrases:
                    if phrase in response_lower:
                        return {
                            "status": "not_found",
                            "message": "Информация не найдена",
                            "results": [{
                                "content": response_text,
                                "section": "Извлеченная информация",
                                "sources": sources
                            }]
                        }
                
                # Форматируем успешный ответ
                return {
                    "status": "success",
                    "results": [{
                        "content": response_text,
                        "section": "Извлеченная информация",
                        "sources": sources,
                        "query": answer.get("query", "")
                    }]
                }
            else:
                response_text = str(answer)
        elif "result" in response:
            response_text = response["result"]
        else:
            return {"status": "error", "message": "Неизвестный формат ответа"}
    else:
        response_text = str(response)
    
    # Для простых ответов (не из RAG chain)
    return {
        "status": "success",
        "results": [{
            "content": response_text,
            "section": "Ответ"
        }]
    }

if __name__ == '__main__':
    logger.info("Запуск веб-сервера...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)