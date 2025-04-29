from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from auth import GigaChatAuth
from document_analyzer import DocumentAnalyzer
import re
import sys
import threading
import time
import traceback
import ssl
import urllib3
import httpx

# ФУНДАМЕНТАЛЬНОЕ отключение проверки SSL-сертификатов на всех уровнях
# 1. Для стандартной библиотеки Python
import ssl
original_context = ssl.create_default_context
def patched_create_default_context(*args, **kwargs):
    ctx = original_context(*args, **kwargs)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx
ssl.create_default_context = patched_create_default_context
ssl._create_default_https_context = ssl._create_unverified_context

# 2. Для urllib3, используемого в requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 3. Для httpx, используемого в gigachat
os.environ["HTTPX_VERIFY"] = "False"
original_transport = httpx.HTTPTransport
def patched_transport(*args, **kwargs):
    kwargs['verify'] = False
    return original_transport(*args, **kwargs)
httpx.HTTPTransport = patched_transport

# 4. Для асинхронного httpx
if hasattr(httpx, 'AsyncHTTPTransport'):
    original_async_transport = httpx.AsyncHTTPTransport
    def patched_async_transport(*args, **kwargs):
        kwargs['verify'] = False
        return original_async_transport(*args, **kwargs)
    httpx.AsyncHTTPTransport = patched_async_transport

# 5. Патчим TLS-соединения на самом низком уровне
original_wrap_socket = ssl.SSLContext.wrap_socket
def patched_wrap_socket(self, *args, **kwargs):
    kwargs['server_hostname'] = None
    return original_wrap_socket(self, *args, **kwargs)
ssl.SSLContext.wrap_socket = patched_wrap_socket

# 6. Отключаем проверку SSL для gigachat через переменные окружения
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

app = Flask(__name__)

# Загружаем переменные окружения
load_dotenv()

# Инициализируем GigaChat клиент
client_id = os.getenv("GIGACHAT_CLIENT_ID")
client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")
client = GigaChatAuth(client_id, client_secret, disable_ssl_verification=True)

# Инициализируем анализатор документов с GigaChatEmbeddings
print("Используются эмбеддинги GigaChat для векторного представления текста")
analyzer = DocumentAnalyzer(client)

# Читаем документы при запуске
try:
    print("Запуск индексации документов...")
    analyzer.read_documents()
    print("Индексация завершена!")
except Exception as e:
    print(f"Ошибка при индексации документов: {str(e)}")
    print("Приложение запущено без индексации документов. Функциональность может быть ограничена.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Получаем запрос от пользователя
    query = request.form.get('query', '')
    
    if not query:
        print("Получен пустой запрос")
        return jsonify({
            "status": "error",
            "message": "Пустой запрос"
        }), 400
    
    print(f"Получен запрос: {query}")
    
    # Проверяем наличие префикса "Требуется информация"
    has_prefix = query.lower().startswith('требуется информация')
    
    # Если нет префикса, отправляем запрос напрямую в GigaChat без анализа документов
    if not has_prefix:
        try:
            print("Запрос без префикса, перенаправляю напрямую в GigaChat")
            # Отправляем запрос напрямую в GigaChat
            response = client.chat_completion(query)
            return jsonify({
                "status": "success",
                "results": [{
                    "content": response,
                    "section": "Ответ GigaChat",
                    "document": "Общая информация"
                }]
            })
        except Exception as e:
            print(f"Ошибка при запросе к GigaChat: {str(e)}")
            traceback.print_exc()
            return jsonify({
                "status": "error",
                "message": f"Ошибка при обращении к GigaChat: {str(e)}"
            }), 500
    
    # Здесь запрос с префиксом, выполняем RAG анализ документов
    try:
        # Получаем ответ от анализатора с таймаутом 30 секунд
        print("Запуск RAG анализа документов...")
        
        # Устанавливаем максимальное время для анализа
        max_execution_time = 30  # секунд
        start_time = time.time()
        
        result = analyze_documents(query, max_execution_time)
        
        if result["status"] == "timeout":
            return jsonify(result), 408
        
        if result["status"] == "error":
            print(f"Ошибка при анализе: {result['message']}")
            return jsonify(result), 500
        
        # Получаем текст ответа
        response_text = result["results"]
        print(f"Получен ответ от анализатора ({len(response_text) if response_text else 0} символов)")
        
        # Обрабатываем ответ для структурированного отображения
        formatted_response = parse_response(response_text)
        print(f"Ответ обработан, статус: {formatted_response['status']}")
        
        return jsonify(formatted_response)
        
    except Exception as e:
        print(f"Ошибка при обработке запроса: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Ошибка при обработке запроса: {str(e)}"
        }), 500

def analyze_documents(query, max_execution_time=30):
    """Анализирует документы на основе запроса пользователя с таймаутом, используя RAG"""
    print(f"Получен запрос: {query}")
    
    # Очищаем запрос перед отправкой
    query = query.strip()
    if not query:
        return {"status": "error", "message": "Пустой запрос"}
    
    # Инициализируем результат, который будет возвращен, если выполнение займет слишком много времени
    timeout_result = {
        "status": "timeout",
        "message": "Превышено время обработки запроса. Пожалуйста, попробуйте сформулировать вопрос иначе."
    }
    
    result = None
    def process_query():
        nonlocal result
        try:
            print(f"Начало RAG анализа документов в отдельном потоке")
            
            # Используем глобальный экземпляр analyzer, который уже инициализирован
            # с правильными учетными данными GigaChat
            global analyzer
            
            # Проверяем, что analyzer инициализирован
            if not analyzer:
                print("Ошибка: Анализатор документов не инициализирован")
                result = {"status": "error", "message": "Ошибка сервера: анализатор документов не инициализирован"}
                return
            
            try:
                print(f"Отправка запроса к RAG: {query}")
                response = analyzer.analyze_documents(query)
                print(f"Получен ответ от RAG: {response[:200]}...")
                
                result = {
                    "status": "success",
                    "results": response
                }
                
            except Exception as e:
                print(f"Ошибка при выполнении анализа: {str(e)}")
                traceback.print_exc()
                result = {"status": "error", "message": f"Ошибка при анализе документов: {str(e)}"}
        except Exception as outer_e:
            print(f"Критическая ошибка: {str(outer_e)}")
            traceback.print_exc()
            result = {"status": "error", "message": f"Критическая ошибка при обработке запроса: {str(outer_e)}"}
    
    # Запускаем обработку в отдельном потоке
    thread = threading.Thread(target=process_query)
    thread.daemon = True
    thread.start()
    
    # Ждем завершения потока определенное время
    thread.join(timeout=max_execution_time)
    
    # Если поток все еще активен, возвращаем результат таймаута
    if thread.is_alive():
        print(f"Превышено время выполнения запроса ({max_execution_time} сек)")
        return timeout_result
    
    if result is None:
        print("Ошибка: результат не установлен")
        return {"status": "error", "message": "Ошибка обработки запроса"}
    
    return result

def parse_response(response):
    """Парсит ответ для структурированного отображения"""
    if not response:
        return {"status": "error", "message": "Пустой ответ от анализатора"}
    
    print(f"Парсинг ответа: {response}")
    
    # Проверка на отсутствие информации
    not_found_phrases = [
        "не найдена",
        "не найдено",
        "не содержится",
        "отсутствует",
        "нет информации",
        "информация отсутствует",
        "не удалось найти"
    ]
    
    for phrase in not_found_phrases:
        if phrase in response.lower():
            print(f"Обнаружена фраза отсутствия информации: '{phrase}'")
            return {
                "status": "not_found",
                "message": "Информация по вашему запросу не найдена в документах",
                "results": response
            }
    
    # Шаблоны для извлечения страницы, раздела и документа (усиливаем)
    page_pattern = r'(?:на\s+)?странице?[\s:]+(\d+)'
    section_pattern = r'в разделе [«"]?(.*?)[»"]?[:\.]'
    
    # Расширенные шаблоны для поиска документов
    doc_patterns = [
        r'в\s+(?:документе|файле)[:\s]+[«"]?([^«".,;]+)[»"]?',
        r'из (?:документа|файла)[:\s]+[«"]?([^«".,;]+)[»"]?',
        r'найдено в[:\s]+[«"]?([^«".,;]+)[»"]?',
        r'согласно[:\s]+[«"]?([^«".,;]+)[»"]?',
        r'в\s+[«"]([^«"]+)[»"]',
        r'документ[^:]*?[:\s]+[«"]?([^«".,;]+)[»"]?'
    ]
    
    # Шаблон для поиска названий файлов с расширениями
    file_ext_pattern = r'\b([a-zA-Zа-яА-Я0-9_\-\s]+\.(pdf|doc|docx|xls|xlsx|txt))\b'
    
    # Разбиваем ответ на разделы
    section_markers = [
        "Информация найдена",
        "По вашему запросу",
        "Согласно документу",
        "В документе",
        "Страница",
        "На странице",
        "Раздел",
        "В разделе",
        "Документ:"
    ]
    
    # Разделяем по маркерам, создавая список разделов
    sections = []
    current_text = response
    
    for marker in section_markers:
        parts = current_text.split(marker)
        if len(parts) > 1:
            for i in range(1, len(parts)):
                section_text = marker + parts[i]
                if section_text.strip():
                    sections.append(section_text.strip())
    
    # Если не удалось разделить на секции, используем весь текст как одну секцию
    if not sections:
        sections = [response]
    
    print(f"Найдено {len(sections)} разделов в ответе")
    
    results = []
    
    for i, section in enumerate(sections):
        print(f"Обработка раздела {i+1}: {section[:100]}...")
        
        page_match = re.search(page_pattern, section, re.IGNORECASE)
        section_match = re.search(section_pattern, section, re.IGNORECASE)
        
        # Пробуем найти имя документа в секции (усиленный поиск)
        doc_name = None
        
        # Сначала ищем по расширенным шаблонам
        for pattern in doc_patterns:
            doc_match = re.search(pattern, section, re.IGNORECASE)
            if doc_match:
                doc_name = doc_match.group(1).strip()
                print(f"Найдено имя документа по шаблону '{pattern}': '{doc_name}'")
                break
        
        # Если имя документа не найдено, ищем файлы с расширениями
        if not doc_name:
            file_match = re.search(file_ext_pattern, section, re.IGNORECASE)
            if file_match:
                doc_name = file_match.group(1).strip()
                print(f"Найдено имя документа по расширению: '{doc_name}'")
        
        # Ищем в окрестностях ключевых слов
        if not doc_name:
            for keyword in ["документ", "файл", "из", "в", "источник"]:
                if keyword in section.lower():
                    # Берем 100 символов после слова и проверяем на подходящие паттерны
                    pos = section.lower().find(keyword)
                    substring = section[pos:pos+100]
                    file_match = re.search(file_ext_pattern, substring, re.IGNORECASE)
                    if file_match:
                        doc_name = file_match.group(1).strip()
                        print(f"Найдено имя документа по ключевому слову '{keyword}': '{doc_name}'")
                        break
        
        page = page_match.group(1) if page_match else "1"  # Используем "1" вместо None для страницы
        section_name = section_match.group(1) if section_match else "Извлечённая информация"
        
        # Если имя документа не найдено, стараемся определить из контекста
        if not doc_name:
            # Пытаемся найти формат документа в тексте
            if 'excel' in section.lower() or 'таблиц' in section.lower() or '.xls' in section.lower():
                doc_name = "Таблица Excel"
            elif 'word' in section.lower() or '.doc' in section.lower():
                doc_name = "Документ Word"
            elif 'pdf' in section.lower() or '.pdf' in section.lower():
                doc_name = "Документ PDF"
            elif 'текстов' in section.lower() or '.txt' in section.lower():
                doc_name = "Текстовый файл"
            else:
                # Если ничего не нашли, извлекаем из названия раздела
                doc_name = section_name.split(',')[0] if ',' in section_name else "Документ"
        
        # Если имя документа является частью названия раздела, разделяем их
        if section_name and section_name.startswith(doc_name) and ',' in section_name:
            section_name = section_name.split(',', 1)[1].strip()
        
        # Очищаем текст от служебных меток
        content = section
        if section_match:
            # Берем текст после "в разделе X:"
            content_parts = re.split(section_pattern, section, re.IGNORECASE)
            if len(content_parts) > 1:
                content = content_parts[-1].strip()
                
        # Если контент слишком короткий или это просто часть маркера, используем весь раздел
        if len(content) < 20 or content.strip() in section_markers:
            content = section
        
        # Логирование найденной информации
        print(f"  Результат для раздела {i+1}:")
        print(f"  - Документ: {doc_name}")
        print(f"  - Раздел: {section_name}")
        print(f"  - Страница: {page}")
        print(f"  - Длина контента: {len(content)}")
            
        # Формируем результат
        result_item = {
            "content": content.strip(),
            "page": page,  # Всегда указываем страницу
            "section": section_name,
            "document": doc_name or "Неизвестный документ"  # Всегда указываем документ
        }
            
        results.append(result_item)
    
    # Если не удалось разобрать структурированно, возвращаем весь текст
    if not results:
        print("Не удалось извлечь структурированную информацию, возвращаем весь текст")
        
        # Ищем имя файла в тексте
        file_match = re.search(file_ext_pattern, response, re.IGNORECASE)
        doc_name = file_match.group(1) if file_match else "Документ с найденной информацией"
        
        return {
            "status": "success",
            "results": [{
                "content": response.strip(),
                "section": "Полученная информация",
                "document": doc_name,
                "page": "1"
            }]
        }
    
    return {
        "status": "success",
        "results": results
    }

if __name__ == '__main__':
    print("Запуск веб-интерфейса анализатора документов с GigaChat RAG...")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True) 