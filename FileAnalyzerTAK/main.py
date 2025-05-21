import os
import ssl
import argparse
import json
from datetime import datetime
from typing import Dict, Optional

# Парсим аргументы командной строки
parser = argparse.ArgumentParser(description='Запуск анализатора документов в консоли')
parser.add_argument('--offline', action='store_true', help='Запуск в полностью офлайн-режиме')
parser.add_argument('--no-vector-db', action='store_true', help='Запуск без использования векторной базы данных')
parser.add_argument('--location', choices=['home', 'work'], help='Местоположение (дома или на работе)')
args = parser.parse_args()

# Загружаем настройки из .env
from dotenv import load_dotenv
load_dotenv()

# Проверяем, нужно ли запускать в офлайн-режиме
OFFLINE_MODE = args.offline or os.getenv('OFFLINE_MODE', 'False').lower() == 'true'
if OFFLINE_MODE:
    print("Запуск в офлайн-режиме: все сетевые запросы отключены")
    os.environ['OFFLINE_MODE'] = 'true'

# Проверяем, нужно ли отключить векторную базу данных
DISABLE_VECTOR_DB = args.no_vector_db or os.getenv('DISABLE_VECTOR_DB', 'False').lower() == 'true'
if DISABLE_VECTOR_DB:
    print("Запуск без использования векторной базы данных")
    os.environ['DISABLE_VECTOR_DB'] = 'true'
    # Разрешаем продолжение работы без векторного хранилища
    os.environ['ALLOW_NO_VECTOR_DB'] = 'true'

# Отключаем проверку SSL и телеметрию
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CHROMA_TELEMETRY_ENABLED'] = 'False'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGCHAIN_TRACING'] = 'false'
os.environ['LANGCHAIN_SESSION'] = 'false'
os.environ['LANGCHAIN_ENDPOINT'] = ''
os.environ['LANGCHAIN_API_KEY'] = ''
os.environ['LANGCHAIN_PROJECT'] = ''

# Устанавливаем местоположение из командной строки, если задано
if args.location:
    os.environ['WORK_LOCATION'] = args.location
    print(f"Установлено местоположение из командной строки: {args.location}")

# Определяем местоположение - дома или на работе
work_location = os.getenv("WORK_LOCATION", "home").lower()
print(f"Режим работы: {'на работе' if work_location == 'work' else 'дома'}")

# Принудительно запрещаем все сетевые запросы в офлайн-режиме
if OFFLINE_MODE:
    os.environ['NO_PROXY'] = '*'
    os.environ['HTTP_PROXY'] = 'http://localhost:1'
    os.environ['HTTPS_PROXY'] = 'http://localhost:1'
else:
    # Если заданы реальные прокси в .env, используем их
    http_proxy = os.getenv('HTTP_PROXY')
    https_proxy = os.getenv('HTTPS_PROXY')
    if http_proxy and https_proxy:
        print(f"Используются пользовательские настройки прокси")

ssl._create_default_https_context = ssl._create_unverified_context

from auth import GigaChatAuth
from document_analyzer import DocumentAnalyzer

def print_welcome_message():
    """Вывод приветственного сообщения с информацией о возможностях"""
    print("\n" + "="*80)
    print("Добро пожаловать в AI-ассистент для анализа документов!")
    print("="*80)
    
    print("\n📚 Поддерживаемые форматы документов:")
    print("  • PDF документы (.pdf) - с сохранением структуры и метаданных")
    print("  • Word документы (.docx) - включая таблицы и форматирование")
    print("  • Excel файлы (.xlsx, .xls) - с обработкой всех листов")
    print("  • Текстовые файлы (.txt) - с поддержкой кодировок")
    print("  • HTML страницы (.html) - с сохранением структуры")
    print("  • PowerPoint презентации (.pptx) - включая слайды и заметки")
    
    print("\n🔍 Возможности системы:")
    print("  • Мультиагентная обработка запросов")
    print("  • Умный поиск и анализ информации")
    print("  • Структурированные и проверенные ответы")
    print("  • Кэширование результатов для быстрых ответов")
    print("  • Поддержка многоязычных документов")
    print("  • Извлечение метаданных и структуры")
    
    print("\n💡 Как задавать вопросы:")
    print("  1. Начните запрос с 'Требуется информация'")
    print("  2. Будьте конкретны в формулировке")
    print("  3. При необходимости уточняйте контекст")
    print("  4. Используйте уточняющие вопросы")
    
    print("\n⚙️ Дополнительные команды:")
    print("  • 'очистить кэш' - очистка кэша ответов")
    print("  • 'статистика' - информация о загруженных документах")
    print("  • 'выход' или 'exit' - завершение работы")
    print("\n" + "="*80 + "\n")

def get_document_stats(analyzer) -> Dict:
    """Получение статистики по документам"""
    try:
        stats = {
            "total_documents": 0,
            "by_type": {},
            "total_sections": 0,
            "last_update": None
        }
        
        if not os.path.exists(analyzer.documents_dir):
            return stats
            
        for filename in os.listdir(analyzer.documents_dir):
            file_path = os.path.join(analyzer.documents_dir, filename)
            if os.path.isfile(file_path):
                stats["total_documents"] += 1
                ext = os.path.splitext(filename)[1].lower()
                stats["by_type"][ext] = stats["by_type"].get(ext, 0) + 1
                
                # Обновляем время последнего изменения
                mtime = os.path.getmtime(file_path)
                if not stats["last_update"] or mtime > stats["last_update"]:
                    stats["last_update"] = mtime
        
        # Получаем количество секций из векторного хранилища
        if analyzer.vector_store:
            stats["total_sections"] = len(analyzer.vector_store.get()["ids"])
        
        return stats
        
    except Exception as e:
        print(f"Ошибка получения статистики: {str(e)}")
        return {}

def print_stats(stats: Dict):
    """Вывод статистики в красивом формате"""
    if not stats:
        print("Статистика недоступна")
        return
        
    print("\n📊 Статистика документов:")
    print(f"  Всего документов: {stats['total_documents']}")
    print(f"  Всего секций: {stats['total_sections']}")
    
    if stats["last_update"]:
        last_update = datetime.fromtimestamp(stats["last_update"])
        print(f"  Последнее обновление: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n  По типам файлов:")
    for ext, count in stats["by_type"].items():
        print(f"    • {ext}: {count}")

def main():
    # Получаем учетные данные GigaChat из переменных окружения
    client_id = os.getenv("GIGACHAT_CLIENT_ID")
    client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("Ошибка: Не найдены учетные данные GigaChat в файле .env")
        print("Пожалуйста, создайте файл .env с переменными GIGACHAT_CLIENT_ID и GIGACHAT_CLIENT_SECRET")
        return
    
    # Инициализируем клиент GigaChat
    client = GigaChatAuth(client_id, client_secret, verify_ssl=False)
    
    # Инициализируем анализатор документов
    analyzer = DocumentAnalyzer(client)
    
    # Принудительно перечитываем документы
    print("Запуск индексации документов с улучшенными алгоритмами обработки...")
    analyzer.read_documents()
    
    # Выводим приветственное сообщение
    print_welcome_message()
    
    while True:
        try:
            user_input = input("\nВаш запрос: ").strip()
            
            if user_input.lower() in ["выход", "exit"]:
                print("До свидания!")
                break
                
            elif user_input.lower() == "очистить кэш":
                analyzer.clear_cache()
                print("Кэш успешно очищен")
                continue
                
            elif user_input.lower() == "статистика":
                stats = get_document_stats(analyzer)
                print_stats(stats)
                continue
            
            if user_input.lower().startswith("требуется информация"):
                response = analyzer.analyze_documents(user_input)
                
                if "error" in response:
                    print(f"\n❌ Ошибка: {response['error']}")
                else:
                    answer = response["answer"]
                    print("\n" + "="*80)
                    print("📝 Ответ:")
                    print("="*80)
                    print(answer["result"])
                    
                    if answer.get("research"):
                        print("\n🔍 Анализ:")
                        print(answer["research"])
                    
                    if answer.get("validation"):
                        print("\n✅ Проверка:")
                        print(answer["validation"])
                    
                    print("\n" + "="*80)
            else:
                print("Запрос должен начинаться с 'Требуется информация'")
                
        except KeyboardInterrupt:
            print("\nПрограмма прервана пользователем")
            break
        except Exception as e:
            print(f"\n❌ Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    main() 