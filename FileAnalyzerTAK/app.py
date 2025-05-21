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
from flask import Flask, render_template, request, jsonify, send_file, Response
from dotenv import load_dotenv
from auth import GigaChatAuth
from document_analyzer import DocumentAnalyzer
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import json
from flask_socketio import SocketIO, emit
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
import queue
import uuid
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.utils
from werkzeug.utils import secure_filename

logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('chromadb.telemetry.posthog').setLevel(logging.CRITICAL)
logging.getLogger('backoff').setLevel(logging.CRITICAL)
logging.getLogger('httpx').setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DocumentAnalyzerApp')

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Очередь для фоновых задач
task_queue = queue.Queue()
task_results = {}

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

# Инициализация приложения
try:
    logger.info("Инициализация GigaChat клиента...")
    client_id = os.getenv("GIGACHAT_CLIENT_ID")
    client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        raise ValueError("Не найдены учетные данные GigaChat (GIGACHAT_CLIENT_ID или GIGACHAT_CLIENT_SECRET)")
    
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
    
    # Инициализируем клиент GigaChat
    client = GigaChatAuth(client_id, client_secret, verify_ssl=False)
    logger.info("GigaChat клиент успешно инициализирован")
    
    # Очищаем папку chroma перед запуском
    clean_chroma_dir()
    
    # Инициализируем анализатор документов
    logger.info("Инициализация DocumentAnalyzer...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    documents_path = os.path.join(current_dir, "documents")
    analyzer = DocumentAnalyzer(client, documents_dir=documents_path)
    
    # Убираем лишний вызов индексации, так как она уже выполнена в конструкторе
    logger.info("DocumentAnalyzer успешно инициализирован")
    
except Exception as e:
    logger.error(f"Ошибка инициализации: {str(e)}")
    raise

def process_background_task(task_id: str, task_type: str, data: Dict):
    """Обработка фоновых задач"""
    try:
        if task_type == "analyze_document":
            # Анализ документа и создание графа знаний
            doc_path = data.get("path")
            if doc_path and analyzer:
                # Создаем граф знаний
                G = nx.Graph()
                
                # Извлекаем ключевые концепции и связи
                concepts = analyzer.extract_concepts(doc_path)
                for concept in concepts:
                    G.add_node(concept["id"], 
                             label=concept["text"],
                             type=concept["type"])
                    
                    for relation in concept.get("relations", []):
                        G.add_edge(concept["id"],
                                 relation["target"],
                                 label=relation["type"])
                
                # Сохраняем результаты
                task_results[task_id] = {
                    "status": "completed",
                    "graph": nx.node_link_data(G),
                    "concepts": concepts
                }
                
        elif task_type == "visualize_embeddings":
            # Визуализация эмбеддингов документов
            if analyzer and analyzer.vector_store:
                # Получаем все эмбеддинги
                embeddings = analyzer.vector_store.get()["embeddings"]
                texts = analyzer.vector_store.get()["documents"]
                
                # Уменьшаем размерность
                tsne = TSNE(n_components=2, random_state=42)
                coords = tsne.fit_transform(embeddings)
                
                # Создаем интерактивный график
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    mode='markers+text',
                    text=[doc.metadata.get("source", "") for doc in texts],
                    hovertext=[doc.page_content[:100] + "..." for doc in texts],
                    marker=dict(size=10)
                ))
                
                # Сохраняем результаты
                task_results[task_id] = {
                    "status": "completed",
                    "plot": json.loads(fig.to_json())
                }
                
    except Exception as e:
        logger.error(f"Ошибка обработки фоновой задачи: {str(e)}")
        task_results[task_id] = {
            "status": "error",
            "error": str(e)
        }

def background_worker():
    """Фоновый обработчик задач"""
    while True:
        try:
            task_id, task_type, data = task_queue.get()
            process_background_task(task_id, task_type, data)
            socketio.emit('task_update', {
                'task_id': task_id,
                'status': task_results[task_id]['status']
            })
        except Exception as e:
            logger.error(f"Ошибка в фоновом обработчике: {str(e)}")
        finally:
            task_queue.task_done()

# Запускаем фоновый обработчик
worker_thread = threading.Thread(target=background_worker, daemon=True)
worker_thread.start()

@app.route('/')
def index():
    """Главная страница с интерактивным интерфейсом"""
    return render_template('index.html')

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Получение списка документов"""
    try:
        documents = []
        for filename in os.listdir(analyzer.documents_dir):
            file_path = os.path.join(analyzer.documents_dir, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                doc_type = os.path.splitext(filename)[1].lower().lstrip('.')
                documents.append({
                    "name": filename,
                    "type": doc_type,
                    "size": stat.st_size,
                    "created": stat.st_ctime,
                    "modified": stat.st_mtime
                })
        
        return jsonify({
            "status": "success",
            "documents": documents
        })
    except Exception as e:
        logger.error(f"Ошибка получения списка документов: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_document():
    """API для загрузки документов"""
    try:
        if 'file' not in request.files:
            return jsonify({
                "status": "error",
                "error": "Файл не найден"
            }), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "status": "error",
                "error": "Файл не выбран"
            }), 400
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(analyzer.documents_dir, filename)
            
            # Проверяем, существует ли файл
            if os.path.exists(file_path):
                return jsonify({
                    "status": "error",
                    "error": f"Файл {filename} уже существует"
                }), 400
            
            # Сохраняем файл
            file.save(file_path)
            
            # Добавляем документ в анализатор
            if analyzer.add_document(file_path):
                # Запускаем фоновый анализ
                task_id = str(uuid.uuid4())
                task_queue.put((
                    task_id,
                    "analyze_document",
                    {"path": file_path}
                ))
                
                return jsonify({
                    "status": "success",
                    "message": "Документ успешно загружен",
                    "task_id": task_id
                })
            else:
                # Удаляем файл, если не удалось добавить в анализатор
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"Ошибка удаления файла {filename}: {str(e)}")
                
                return jsonify({
                    "status": "error",
                    "error": "Ошибка добавления документа"
                }), 500
        else:
            return jsonify({
                "status": "error",
                "error": "Неподдерживаемый формат файла"
            }), 400
                
    except Exception as e:
        logger.error(f"Ошибка загрузки документа: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/task/<task_id>', methods=['GET'])
def get_task_status(task_id: str):
    """Получение статуса фоновой задачи"""
    if task_id in task_results:
        return jsonify(task_results[task_id])
    return jsonify({
        "status": "pending"
    })

@app.route('/api/visualize/embeddings', methods=['GET'])
def visualize_embeddings():
    """Визуализация эмбеддингов документов"""
    try:
        task_id = str(uuid.uuid4())
        task_queue.put((
            task_id,
            "visualize_embeddings",
            {}
        ))
        return jsonify({
            "status": "success",
            "task_id": task_id
        })
    except Exception as e:
        logger.error(f"Ошибка создания визуализации: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/documents/<filename>', methods=['GET'])
def get_document(filename: str):
    """Получение содержимого документа с подсветкой"""
    try:
        file_path = os.path.join(analyzer.documents_dir, secure_filename(filename))
        if not os.path.exists(file_path):
            return jsonify({
                "status": "error",
                "error": "Документ не найден"
            }), 404
            
        # Получаем структуру документа
        doc_type = os.path.splitext(filename)[1].lower().lstrip('.')
        structure = analyzer.extract_document_structure(file_path, doc_type)
        
        return jsonify({
            "status": "success",
            "document": {
                "name": filename,
                "type": doc_type,
                "structure": structure,
                "metadata": structure.get("metadata", {})
            }
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения документа: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/graph/<filename>', methods=['GET'])
def get_document_graph(filename: str):
    """Получение графа знаний для документа"""
    try:
        file_path = os.path.join(analyzer.documents_dir, secure_filename(filename))
        if not os.path.exists(file_path):
            return jsonify({
                "status": "error",
                "error": "Документ не найден"
            }), 404
            
        # Создаем граф знаний
        G = nx.Graph()
        concepts = analyzer.extract_concepts(file_path)
        
        for concept in concepts:
            G.add_node(concept["id"],
                      label=concept["text"],
                      type=concept["type"])
            
            for relation in concept.get("relations", []):
                G.add_edge(concept["id"],
                          relation["target"],
                          label=relation["type"])
        
        # Создаем визуализацию
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=1500, font_size=10, font_weight='bold')
        
        # Сохраняем в буфер
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return send_file(
            buf,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'graph_{filename}.png'
        )
        
    except Exception as e:
        logger.error(f"Ошибка создания графа: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """API для задавания вопросов по документам"""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                "status": "error",
                "error": "Вопрос не указан"
            }), 400
            
        question = data['question']
        document_name = data.get('document')  # Опционально - конкретный документ
        
        # Получаем ответ от RAG-цепи
        if document_name:
            # Если указан конкретный документ
            file_path = os.path.join(analyzer.documents_dir, secure_filename(document_name))
            if not os.path.exists(file_path):
                return jsonify({
                    "status": "error",
                    "error": "Документ не найден"
                }), 404
            answer = analyzer.ask_question(question, file_path)
        else:
            # По всем документам
            answer = analyzer.ask_question(question)
            
        return jsonify({
            "status": "success",
            "answer": answer
        })
        
    except Exception as e:
        logger.error(f"Ошибка обработки вопроса: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

def allowed_file(filename: str) -> bool:
    """Проверка допустимых расширений файлов"""
    ALLOWED_EXTENSIONS = {
        'pdf', 'docx', 'doc', 'xlsx', 'xls',
        'txt', 'html', 'pptx', 'ppt'
    }
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@socketio.on('connect')
def handle_connect():
    """Обработка подключения WebSocket"""
    emit('connection_response', {'data': 'Connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Обработка отключения WebSocket"""
    pass

if __name__ == '__main__':
    # Создаем папку для логов
    os.makedirs('logs', exist_ok=True)
    
    # Запускаем Flask-приложение с поддержкой WebSocket
    socketio.run(
        app,
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    )