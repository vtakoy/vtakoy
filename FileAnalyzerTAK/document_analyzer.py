"""
Новая реализация DocumentAnalyzer на основе модулей gigachat_rag_example.py и gigachat_advanced_rag.py
"""

from gigachat_langchain import GigaChatLangchain
from langchain_core.documents import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
import os
import time
import logging
from datetime import datetime
import pandas as pd
from docx import Document
import re
from typing import List, Dict, Any, Optional
import ssl
import urllib3
import httpx
import shutil

# Отключаем проверку SSL для всех компонентов
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["HTTPX_VERIFY"] = "False"

class Logger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self._ensure_log_dir()
        self._setup_logging()
        
    def _ensure_log_dir(self):
        """Создает директорию для логов, если она не существует"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
    def _setup_logging(self):
        """Настраивает логирование"""
        log_file = os.path.join(self.log_dir, f"document_analyzer_{datetime.now().strftime('%Y%m%d')}.log")
        
        # Настраиваем форматирование для файла
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Настраиваем форматирование для консоли
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        
        # Настраиваем корневой логгер
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, console_handler]
        )
        
    def log_operation(self, operation: str, details: str = ""):
        """Логирует операцию"""
        logging.info(f"Операция: {operation}")
        if details:
            logging.info(f"Детали: {details}")
            
    def log_error(self, error: str, details: str = ""):
        """Логирует ошибку"""
        logging.error(f"Ошибка: {error}")
        if details:
            logging.error(f"Детали: {details}")
            
    def log_query(self, query: str, response: str):
        """Логирует запрос и ответ"""
        print("\n=== Запрос пользователя ===")
        print(query)
        print("\n=== Ответ системы ===")
        print(response)
        print("=====================\n")
        
        # Для файла логов сохраняем полную информацию
        logging.info("=== Запрос пользователя ===")
        logging.info(query)
        logging.info("=== Ответ системы ===")
        logging.info(response)
        logging.info("=====================")

class DocumentAnalyzer:
    """Класс для анализа документов и ответов на вопросы с использованием RAG"""
    
    def __init__(self, client):
        """
        Инициализация анализатора документов
        
        Args:
            client: Объект GigaChatAuth с учетными данными для аутентификации
        """
        self.client = client  # Сохраняем клиент для использования
        
        # Получаем токен доступа из клиента
        self.token = self.client.auth_token
        if not self.token:
            self.client.get_token()
            self.token = self.client.auth_token
            
        # Инициализация директорий
        self.documents_dir = "documents"
        if not os.path.exists(self.documents_dir):
            os.makedirs(self.documents_dir)
            
        # Инициализация логгера
        self.logger = Logger()
        
        # Инициализация GigaChat через langchain с использованием токена
        try:
            from langchain_gigachat.embeddings import GigaChatEmbeddings
            from langchain_gigachat.chat_models import GigaChat
            
            # Создаем экземпляры с использованием токена
            self.embeddings = GigaChatEmbeddings(
                credentials=self.token,  # Используем полученный токен
                verify_ssl_certs=False
            )
            
            self.llm = GigaChat(
                credentials=self.token,  # Используем полученный токен
                verify_ssl_certs=False,
                scope="GIGACHAT_API_PERS"
            )
            
            self.logger.log_operation("Инициализация LangChain объектов", "Успешно")
            
        except Exception as e:
            self.logger.log_error("Ошибка при инициализации LangChain объектов", str(e))
            print(f"ПРЕДУПРЕЖДЕНИЕ: {str(e)}")
            # Будем использовать метод через GigaChatLangchain как запасной вариант
            giga = GigaChatLangchain(
                client_id=client.client_id,
                client_secret=client.client_secret,
                verify_ssl=False,
                scope="GIGACHAT_API_PERS"
            )
            
            # Инициализация Langchain объектов через запасной метод
            self.embeddings = giga.get_embeddings()
            self.llm = giga.get_chat_model()
            
        self.qa_chain = None
        self.rag_chain = None
        self.vector_store = None
        
        self.logger.log_operation("Инициализация анализатора документов", "Выполнена успешно")
    
    def _ensure_documents_dir(self):
        """Создает директорию для документов, если она не существует"""
        if not os.path.exists(self.documents_dir):
            os.makedirs(self.documents_dir)
            self.logger.log_operation("Создание директории документов", f"Создана директория: {self.documents_dir}")
    
    def read_documents(self, directory: str = "documents") -> None:
        """Читает все документы из указанной директории и создает векторное хранилище"""
        try:
            self.logger.log_operation(f"Начало чтения документов из директории: {directory}")
            
            if not os.path.exists(directory):
                os.makedirs(directory)
                self.logger.log_operation(f"Создана директория: {directory}")
                return
            
            # Загрузка документов
            documents = self._load_all_documents(directory)
            
            if documents:
                # Создаем векторное хранилище
                self.logger.log_operation("Создание векторного хранилища", "Начало")
                self.vector_store = self._create_vector_store(documents)
                
                # Создаем RAG-цепочку
                self.logger.log_operation("Создание RAG-цепочки", "Начало")
                self.rag_chain = self._create_rag_chain(self.vector_store)
                
                self.logger.log_operation("Инициализация завершена", "Успешно")
            else:
                self.logger.log_error("Документы не найдены", "Директория для документов пуста")
            
        except Exception as e:
            self.logger.log_error(f"Ошибка при чтении документов: {str(e)}")
            raise
    
    def _load_all_documents(self, directory):
        """Загружает документы разных типов из указанной директории"""
        all_sections = []
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if filename.endswith('.txt'):
                sections = self._read_txt(file_path)
            elif filename.endswith('.docx'):
                sections = self._read_word(file_path)
            elif filename.endswith('.xlsx') or filename.endswith('.xls'):
                sections = self._read_excel(file_path)
            else:
                self.logger.log_operation("Пропуск файла", f"Неподдерживаемый формат: {filename}")
                continue
            
            all_sections.extend(sections)
        
        # Создаем LangChain документы
        documents = self._create_langchain_documents(all_sections)
        
        # Разделяем документы на чанки для лучшего поиска
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        
        # Разбиваем документы на чанки
        chunked_documents = text_splitter.split_documents(documents)
        self.logger.log_operation("Статистика обработки", 
                                f"Всего секций: {len(documents)}, создано чанков: {len(chunked_documents)}")
        
        return chunked_documents
    
    def _read_txt(self, file_path: str) -> List[Dict[str, Any]]:
        """Читает текстовый файл"""
        sections = []
        
        # Добавляем заголовок документа как первую секцию для улучшения поиска
        file_title = os.path.splitext(os.path.basename(file_path))[0]
        
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Разбиваем текст на параграфы
        paragraphs = text.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Проверяем, является ли параграф заголовком
            is_header = False
            if paragraph.isupper() or paragraph.endswith(':'):
                is_header = True
            # Проверяем начинается ли строка с номера или специальных символов
            elif re.match(r'^(\d+\.|\-|\*|\#)\s+', paragraph):
                is_header = True
            # Проверяем короткие строки (менее 50 символов) с ключевыми словами
            elif len(paragraph) < 50 and any(kw in paragraph.lower() for kw in ["глава", "раздел", "часть"]):
                is_header = True
            
            section_text = paragraph
            section_name = paragraph if is_header else f"Часть {i+1}"
            
            sections.append({
                "text": section_text,
                "metadata": {
                    "source": os.path.basename(file_path),
                    "section": section_name
                }
            })
        
        self.logger.log_operation(f"Чтение TXT", f"Извлечено {len(sections)} секций из {os.path.basename(file_path)}")
        return sections
    
    def _read_word(self, file_path: str) -> List[Dict[str, Any]]:
        """Читает Word файл и возвращает список словарей с данными"""
        try:
            doc = Document(file_path)
            sections = []
            
            # Добавляем имя файла в виде метаданных для всех секций
            filename = os.path.basename(file_path)
            
            # Добавляем заголовок документа как первую секцию для улучшения поиска
            file_title = os.path.splitext(filename)[0]
            
            current_section = {"text": "", "metadata": {"source": filename, "section": "Основной текст"}}
            
            # Проходим по всем абзацам
            for i, paragraph in enumerate(doc.paragraphs):
                if paragraph.text.strip():
                    # Определяем тип раздела
                    if paragraph.style.name.startswith('Heading'):
                        if current_section["text"].strip():
                            sections.append(current_section.copy())
                        current_section = {
                            "text": paragraph.text.strip() + "\n\n",
                            "metadata": {
                                "source": filename,
                                "section": paragraph.text.strip()
                            }
                        }
                    else:
                        current_section["text"] += paragraph.text + "\n\n"
                        
            if current_section["text"].strip():
                sections.append(current_section.copy())
                
            # Проходим по таблицам - добавляем их содержимое
            for table_idx, table in enumerate(doc.tables):
                table_text = f"Таблица {table_idx + 1}:\n\n"
                
                # Получаем данные таблицы в виде строк
                rows_data = []
                for i, row in enumerate(table.rows):
                    cells_data = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cells_data:
                        row_text = " | ".join(cells_data)
                        rows_data.append(row_text)
                
                if len(rows_data) > 0:
                    # Если первая строка короче (возможно, заголовок), обрабатываем особо
                    if len(rows_data) > 1 and len(rows_data[0].split('|')) < len(rows_data[1].split('|')):
                        table_text += f"Заголовок таблицы: {rows_data[0]}\n\n"
                        for row_idx, row in enumerate(rows_data[1:], 1):
                            table_text += f"Строка {row_idx}: {row}\n"
                    else:
                        for row_idx, row in enumerate(rows_data, 1):
                            table_text += f"Строка {row_idx}: {row}\n"
                
                # Добавляем таблицу как отдельную секцию
                if len(rows_data) > 0:
                    table_section = {
                        "text": table_text,
                        "metadata": {
                            "source": filename,
                            "section": f"Таблица {table_idx + 1}"
                        }
                    }
                    sections.append(table_section)
                
            self.logger.log_operation("Успешное чтение Word", f"Извлечено {len(sections)} секций из {filename}")
            return sections
        except Exception as e:
            self.logger.log_error("Ошибка при чтении Word файла", str(e))
            return []
    
    def _read_excel(self, file_path: str) -> List[Dict]:
        """Читает Excel файл и возвращает список словарей с данными"""
        try:
            self.logger.log_operation("Чтение Excel файла", f"Обработка файла: {file_path}")
            df = pd.read_excel(file_path)
            
            sections = []
            filename = os.path.basename(file_path)
            
            # Создаем общий обзор таблицы
            overview = {
                "text": f"Обзор таблицы из файла {filename}:\n\n",
                "metadata": {
                    "source": filename,
                    "section": "Обзор таблицы"
                }
            }
            
            # Добавляем заголовки таблицы
            headers = " | ".join(df.columns)
            overview["text"] += f"Заголовки таблицы: {headers}\n\n"
            
            # Добавляем статистику по таблице
            overview["text"] += f"Количество строк: {len(df)}\n"
            overview["text"] += f"Количество столбцов: {len(df.columns)}\n\n"
            
            sections.append(overview)
            
            # Разбиваем таблицу на секции по 20 строк
            for i in range(0, len(df), 20):
                section_df = df.iloc[i:i+20]
                section_text = f"Данные из таблицы (строки {i+1}-{min(i+20, len(df))}):\n\n"
                
                # Добавляем данные построчно
                for index, row in section_df.iterrows():
                    row_text = " | ".join([str(val) for val in row])
                    section_text += f"Строка {index + 1}: {row_text}\n"
                
                sections.append({
                    "text": section_text,
                    "metadata": {
                        "source": filename,
                        "section": f"Часть таблицы {i//20 + 1}"
                    }
                })
            
            self.logger.log_operation("Успешное чтение Excel", f"Извлечено {len(sections)} секций из {filename}")
            return sections
        except Exception as e:
            self.logger.log_error("Ошибка при чтении Excel файла", str(e))
            return []
    
    def _create_langchain_documents(self, sections: List[Dict]) -> List[LangchainDocument]:
        """Преобразует секции в документы LangChain"""
        documents = []
        for section in sections:
            documents.append(LangchainDocument(
                page_content=section["text"],
                metadata=section["metadata"]
            ))
        return documents
    
    def _create_vector_store(self, documents: List[LangchainDocument]) -> Chroma:
        """Создает векторное хранилище Chroma из документов"""
        try:
            self.logger.log_operation("Создание векторного хранилища Chroma", f"Обработка {len(documents)} документов")
            
            # Удаляем существующую базу Chroma, если она есть
            if os.path.exists("./chroma_db"):
                self.logger.log_operation("Удаление старого хранилища Chroma", "Начало")
                try:
                    shutil.rmtree("./chroma_db")
                    self.logger.log_operation("Удаление старого хранилища Chroma", "Успешно")
                except PermissionError as e:
                    self.logger.log_error("Ошибка доступа к файлу Chroma", str(e))
                    self.logger.log_operation("Повторная попытка удаления через 1 секунду", "")
                    time.sleep(1)  # Добавляем задержку перед повторной попыткой
                    try:
                        shutil.rmtree("./chroma_db")
                        self.logger.log_operation("Удаление старого хранилища Chroma", "Успешно после повторной попытки")
                    except Exception as e:
                        self.logger.log_error("Не удалось удалить старую базу Chroma", str(e))
                        raise
            
            # Добавляем небольшую задержку перед созданием нового хранилища
            time.sleep(0.5)
            
            # Добавляем инструкцию для эмбеддингов согласно документации GigaChat Embeddings
            for doc in documents:
                if not 'instruction' in doc.metadata:
                    doc.metadata['instruction'] = "Дан вопрос, необходимо найти абзац текста с ответом"
            
            # Создаем векторное хранилище Chroma
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
            
            # Сохраняем векторное хранилище
            vector_store.persist()
            
            self.logger.log_operation("Векторное хранилище Chroma создано", "Успешно")
            return vector_store
            
        except Exception as e:
            self.logger.log_error("Ошибка при создании векторного хранилища", str(e))
            
            # Создаем резервное простое хранилище без эмбеддингов, чтобы приложение могло работать
            self.logger.log_operation("Создание резервного хранилища", "Попытка создать простое хранилище")
            try:
                from langchain_community.embeddings import FakeEmbeddings
                dummy_embeddings = FakeEmbeddings(size=1536)  # Размер как у GigaChat embeddings
                
                vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=dummy_embeddings,
                    persist_directory="./chroma_db_fallback"
                )
                vector_store.persist()
                
                self.logger.log_operation("Создано резервное хранилище", "Будет использоваться для базовой функциональности")
                return vector_store
            except Exception as fallback_error:
                self.logger.log_error("Не удалось создать резервное хранилище", str(fallback_error))
                raise e
    
    def _create_rag_chain(self, vector_store: Chroma) -> RetrievalQA:
        """Создает RAG-цепочку для ответов на вопросы"""
        prompt_template = """
        Ты - эксперт по анализу документов. Твоя задача - найти и проанализировать информацию из документов, соответствующую запросу.

        Запрос пользователя: {question}

        Контекст из документов:
        {context}

        Инструкции:
        1. Внимательно проанализируй предоставленный контекст из документов
        2. Найди информацию, которая соответствует запросу пользователя
        3. ВАЖНО: ДАЖЕ ЕСЛИ ИНФОРМАЦИЯ ПРЕДОСТАВЛЕНА ЧАСТИЧНО ИЛИ КОСВЕННО, ИСПОЛЬЗУЙ ЕЁ ДЛЯ ОТВЕТА
        4. Если информация найдена, укажи:
           - Название документа, где найдена информация
           - Точный раздел, где найдена информация
           - Подробный ответ на основе найденной информации
        5. Изложи найденную информацию в структурированном виде, с деталями и пояснениями
        6. Если информация найдена в нескольких документах, объедини ее в целостный ответ

        Формат ответа:
        "Информация найдена в разделе <Наименование раздела> документа <Имя файла>: <Ответ с найденной информацией>"

        ВАЖНО: 
        - Отвечай ТОЛЬКО на основе информации из предоставленных документов
        - ИСПОЛЬЗУЙ ЛЮБУЮ РЕЛЕВАНТНУЮ ИНФОРМАЦИЮ ИЗ КОНТЕКСТА, даже если она не точно соответствует запросу
        - Можешь перефразировать и интерпретировать информацию для лучшего понимания
        - Не добавляй информацию из своих знаний, если её нет в документах
        - Обязательно укажи источник информации (название документа)
        - НИКОГДА не отвечай "Информация не найдена", если в контексте есть хоть какая-то релевантная информация!
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Создаем retriever с инструкцией для улучшения поиска по документам
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 6,  # Увеличиваем количество документов для большего контекста
                "search_type": "similarity"
            }
        )
        
        # Улучшенный retriever с инструкциями
        class InstructionRetriever:
            def __init__(self, base_retriever):
                self.base_retriever = base_retriever
                
            def get_relevant_documents(self, query):
                # Добавляем инструкцию для retrieval-задачи
                instruction = "Дан вопрос, необходимо найти информацию в документах:\n"
                
                # Проверяем, является ли запрос уже структурированным
                if query.lower().startswith("требуется информация"):
                    # Извлекаем основную суть запроса
                    clean_query = query.lower().replace("требуется информация", "").strip()
                    augmented_query = f"{instruction}{clean_query}"
                else:
                    augmented_query = f"{instruction}{query}"
                    
                return self.base_retriever.get_relevant_documents(augmented_query)
        
        # Создаем retriever с инструкциями
        instructed_retriever = InstructionRetriever(retriever)
        
        # Создаем RetrievalQA цепочку
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=instructed_retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    def analyze_documents(self, query: str) -> str:
        """Анализирует документы с помощью RAG и возвращает ответ на запрос"""
        if not query.lower().startswith("требуется информация"):
            self.logger.log_error("Некорректный запрос", "Запрос должен начинаться с 'Требуется информация'")
            return "Запрос должен начинаться с 'Требуется информация'"
            
        try:
            self.logger.log_operation("Анализ запроса", "Начало анализа")
            self.logger.log_query(query, "")  # Логируем запрос отдельно
            
            if not self.vector_store:
                self.logger.log_error("Векторное хранилище не инициализировано", "Сначала прочитайте документы")
                return "Сначала прочитайте документы с помощью метода read_documents()"
            
            # Извлекаем запрос для RAG
            clean_query = query.lower().replace("требуется информация", "").strip()
            self.logger.log_operation("Обработка запроса", f"Ключевые слова: '{clean_query}'")
            
            # Выполняем RAG запрос
            response = self.rag_chain.invoke({"query": clean_query})
            
            # Логируем результат
            result = response.get("result", "Ошибка: не удалось получить ответ")
            self.logger.log_operation("Ответ RAG получен", f"Длина ответа: {len(result)} символов")
            self.logger.log_query(query, result)
            
            return result
            
        except Exception as e:
            self.logger.log_error("Ошибка при анализе документов", str(e))
            return f"Ошибка при анализе документов: {str(e)}"
    
    def add_document(self, file_path: str) -> bool:
        """Добавляет новый документ в директорию для анализа"""
        try:
            if not os.path.exists(file_path):
                self.logger.log_error("Файл не найден", f"Путь: {file_path}")
                return False
                
            filename = os.path.basename(file_path)
            target_path = os.path.join(self.documents_dir, filename)
            
            # Копируем файл в директорию документов
            with open(file_path, 'rb') as src, open(target_path, 'wb') as dst:
                dst.write(src.read())
                
            self.logger.log_operation("Добавление документа", f"Файл {filename} успешно добавлен")
            
            # Перечитываем документы для обновления векторного хранилища
            self.read_documents()
            
            return True
        except Exception as e:
            self.logger.log_error("Ошибка при добавлении документа", str(e))
            return False