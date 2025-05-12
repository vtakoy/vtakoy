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
    
    os.environ.update({
        'GIGACHAT_AUTH_URL': 'https://sm-auth-sd.prom-88-89-apps.ocp-geo.ocp.sigma.sbrf.ru/api/v2/oauth',
        'GIGACHAT_BASE_URL': 'https://gigachat.devices.sberbank.ru/api/v1'
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
    
    not_found_phrases = ["не найдена", "не найдено", "отсутствует"]
    for phrase in not_found_phrases:
        if phrase in response.lower():
            return {"status": "not_found", "message": "Информация не найдена", "results": response}
    
    return {
        "status": "success",
        "results": [{"content": response, "section": "Извлеченная информация"}]
    }

if __name__ == '__main__':
    logger.info("Запуск веб-сервера...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)