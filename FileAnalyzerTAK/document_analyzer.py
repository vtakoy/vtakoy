"""
DocumentAnalyzer - полная реализация с исправлениями ошибок
"""
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
import ssl
import os
import time
import shutil
import base64
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from docx import Document
import re
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain_gigachat.embeddings import GigaChatEmbeddings
from langchain_gigachat.chat_models import GigaChat
from tenacity import retry, stop_after_attempt, wait_fixed

# Конфигурация
persist_directory = "./chroma_db"
collection_name = "documents_collection"

class DocumentAnalyzer:
    """Класс для анализа документов с использованием GigaChat и ChromaDB"""
    
    def __init__(self, client):
        """
        Инициализация анализатора документов
        
        Args:
            client: Объект GigaChatAuth с учетными данными
        """
        self.client = client
        self.token = client.auth_token or client.get_token()
        self.documents_dir = "documents"
        os.makedirs(self.documents_dir, exist_ok=True)
        
        # Инициализация логгера
        self._init_logger()
        
        # Инициализация LLM компонентов
        self._init_llm_components()
        
        self.vector_store = None
        self.rag_chain = None
        self.logger.info("Анализатор документов инициализирован")

    def _init_logger(self):
        """Настройка системы логирования"""
        self.logger = logging.getLogger('DocumentAnalyzer')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Логи в файл
        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler(
            f"logs/document_analyzer_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        
        # Вывод в консоль
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _init_llm_components(self):
        """Инициализация компонентов LangChain"""
        try:
            # Добавляем SSL контекст
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            ssl_context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
            
            credentials = base64.b64encode(
                f"{self.client.client_id}:{self.client.client_secret}".encode()
            ).decode()
            
            self.embeddings = GigaChatEmbeddings(
                credentials=credentials,
                verify_ssl_certs=False,
                scope="GIGACHAT_API_PERS",
                timeout=30,
                ssl_context=ssl_context  # Передаем контекст
            )
            
            self.llm = GigaChat(
                credentials=credentials,
                verify_ssl_certs=False,
                scope="GIGACHAT_API_PERS",
                temperature=0.7,
                max_tokens=1500,
                ssl_context=ssl_context  # Передаем контекст
            )
            
            self.logger.info("Компоненты LangChain инициализированы")
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации LLM: {str(e)}")
            raise

    def read_documents(self, directory: str = None) -> None:
        """Чтение и индексация документов"""
        dir_path = directory or self.documents_dir
        try:
            self.logger.info(f"Начало индексации документов из {dir_path}")
            
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                return
                
            documents = self._load_all_documents(dir_path)
            if documents:
                self.vector_store = self._create_vector_store(documents)
                self.rag_chain = self._create_rag_chain()
                self.logger.info(f"Индексация завершена. Обработано документов: {len(documents)}")
                
        except Exception as e:
            self.logger.error(f"Ошибка индексации документов: {str(e)}")
            raise

    def _load_all_documents(self, directory: str) -> List[LangchainDocument]:
        """Загрузка документов разных форматов"""
        all_sections = []
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            try:
                if filename.endswith('.txt'):
                    sections = self._read_txt(file_path)
                elif filename.endswith('.docx'):
                    sections = self._read_word(file_path)
                elif filename.endswith(('.xlsx', '.xls')):
                    sections = self._read_excel(file_path)
                else:
                    self.logger.warning(f"Неподдерживаемый формат: {filename}")
                    continue
                    
                all_sections.extend(sections)
                self.logger.info(f"Обработан файл: {filename} - секций: {len(sections)}")
                
            except Exception as e:
                self.logger.error(f"Ошибка обработки файла {filename}: {str(e)}")
                continue
        
        # Создание документов LangChain
        documents = [
            LangchainDocument(
                page_content=section["text"],
                metadata=section["metadata"]
            ) for section in all_sections
        ]
        
        # Разделение на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        
        return text_splitter.split_documents(documents)

    def _read_txt(self, file_path: str) -> List[Dict]:
        """Чтение текстового файла"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return [{
            "text": text,
            "metadata": {
                "source": os.path.basename(file_path),
                "section": "Основное содержимое"
            }
        }]

    def _read_word(self, file_path: str) -> List[Dict]:
        """Чтение Word документа"""
        try:
            doc = Document(file_path)
            sections = []
            filename = os.path.basename(file_path)
            
            for para in doc.paragraphs:
                if para.text.strip():
                    sections.append({
                        "text": para.text,
                        "metadata": {
                            "source": filename,
                            "section": para.style.name if para.style.name.startswith('Heading') else "Параграф"
                        }
                    })
                    
            # Обработка таблиц
            for table_idx, table in enumerate(doc.tables, 1):
                table_text = f"Таблица {table_idx}:\n"
                for row_idx, row in enumerate(table.rows, 1):
                    row_text = " | ".join(cell.text for cell in row.cells)
                    table_text += f"Строка {row_idx}: {row_text}\n"
                
                sections.append({
                    "text": table_text,
                    "metadata": {
                        "source": filename,
                        "section": f"Таблица {table_idx}"
                    }
                })
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Ошибка чтения Word файла {file_path}: {str(e)}")
            return []

    def _read_excel(self, file_path: str) -> List[Dict]:
        """Чтение Excel файла"""
        try:
            df = pd.read_excel(file_path)
            filename = os.path.basename(file_path)
            sections = []
            
            # Общая информация
            sections.append({
                "text": f"Заголовки: {', '.join(df.columns)}\n\nПервые строки:\n{df.head().to_string()}",
                "metadata": {
                    "source": filename,
                    "section": "Обзор таблицы"
                }
            })
            
            # Разделение на части
            chunk_size = 20
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                sections.append({
                    "text": f"Данные (строки {i+1}-{i+len(chunk)}):\n{chunk.to_string()}",
                    "metadata": {
                        "source": filename,
                        "section": f"Часть {i//chunk_size + 1}"
                    }
                })
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Ошибка чтения Excel файла {file_path}: {str(e)}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _create_vector_store(self, documents: List[LangchainDocument]) -> Chroma:
        """
        Создание векторного хранилища ChromaDB с использованием нового API (chromadb>=0.4.22)
        """
        try:
            # 1. Очистка предыдущей базы данных
            if os.path.exists(persist_directory):
                self.logger.info("Удаление старой базы Chroma...")
                try:
                    shutil.rmtree(persist_directory, ignore_errors=True)
                    time.sleep(1)
                except Exception as e:
                    self.logger.error(f"Ошибка удаления базы: {str(e)}")
                    raise

            # 2. Создание клиента ChromaDB
            from chromadb import PersistentClient
            chroma_client = PersistentClient(path=persist_directory)

            # 3. Создание векторного хранилища через новый API
            self.logger.info("Создание нового векторного хранилища...")
            vector_store = Chroma(
                client=chroma_client,
                collection_name=collection_name,
                embedding_function=self.embeddings,
            )
            # 4. Индексация документов
            vector_store.add_documents(documents)
            self.logger.info("Векторное хранилище успешно создано и документы проиндексированы")
            return vector_store

        except Exception as e:
            self.logger.error(f"Критическая ошибка при создании хранилища: {str(e)}")
            raise RuntimeError("Не удалось инициализировать векторное хранилище") from e

    def _create_rag_chain(self) -> RetrievalQA:
        """Создание RAG цепочки"""
        prompt_template = """
        Ты - эксперт по анализу документов. Ответь на вопрос на основе предоставленного контекста.

        Вопрос: {question}

        Контекст: {context}

        Требования:
        1. Отвечай только на основе контекста
        2. Указывай источник информации
        3. Будь точным и информативным
        4. Если информация не найдена, так и скажи

        Формат ответа:
        Источник: [документ, раздел]
        Ответ: [развернутый ответ]
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def analyze_documents(self, query: str) -> str:
        """Анализ документов с помощью RAG"""
        if not query.lower().startswith("требуется информация"):
            return "Запрос должен начинаться с 'Требуется информация'"
            
        try:
            clean_query = query.replace("Требуется информация", "").strip()
            self.logger.info(f"Обработка запроса: {clean_query}")
            
            if not self.vector_store:
                raise ValueError("Векторное хранилище не инициализировано")
                
            response = self.rag_chain.invoke({"query": clean_query})
            result = response.get("result", "Не удалось получить ответ")
            
            self.logger.info(f"Запрос обработан. Длина ответа: {len(result)} символов")
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа документов: {str(e)}")
            return f"Ошибка: {str(e)}"

    def add_document(self, file_path: str) -> bool:
        """Добавление нового документа"""
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"Файл не найден: {file_path}")
                return False
                
            target_path = os.path.join(self.documents_dir, os.path.basename(file_path))
            shutil.copy(file_path, target_path)
            
            self.logger.info(f"Документ добавлен: {os.path.basename(file_path)}")
            self.read_documents()  # Переиндексация
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка добавления документа: {str(e)}")
            return False