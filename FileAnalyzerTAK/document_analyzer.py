import os
import shutil
import logging
import base64
import ssl
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from docx import Document
import pandas as pd
from bs4 import BeautifulSoup
from pptx import Presentation
import re
import pytesseract
from pdf2image import convert_from_path
import tabula
from unstructured.partition.auto import partition
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter
)
from langchain_core.documents import Document as LangchainDocument
from langchain_gigachat.embeddings import GigaChatEmbeddings
from langchain_gigachat.chat_models import GigaChat
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential, retry_if_exception_type
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
import time
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
from functools import lru_cache
from langchain_community.cache import InMemoryCache
from langchain_community.callbacks.manager import get_openai_callback
from crewai import Agent, Task, Crew, Process
from cachetools import TTLCache, cached
import networkx as nx
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    TextLoader
)
from chromadb.config import Settings as ChromaSettings
from chromadb import PersistentClient

current_dir = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(current_dir, "chroma")
COLLECTION_NAME = "documents_collection"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CHUNKS_PER_DOC = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_TTL = 3600  # 1 час
MAX_CACHE_SIZE = 1000

# Настройки для OCR
TESSERACT_CONFIG = {
    'lang': 'rus+eng',
    'config': '--oem 3 --psm 6'
}

# Настройки для извлечения таблиц
TABLE_SETTINGS = {
    'lattice': True,
    'stream': True,
    'guess': True,
    'pages': 'all'
}

# Добавляем новые константы для улучшенной обработки
CHUNK_STRATEGIES = {
    'pdf': {
        'chunk_size': 1000,
        'chunk_overlap': 200,
        'separators': ["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
    },
    'docx': {
        'chunk_size': 800,
        'chunk_overlap': 150,
        'separators': ["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
    },
    'excel': {
        'chunk_size': 500,
        'chunk_overlap': 100,
        'separators': ["\n", "|", ",", " ", ""]
    },
    'default': {
        'chunk_size': 800,
        'chunk_overlap': 200,
        'separators': ["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
    }
}

# Настройки для многопроходного поиска
SEARCH_SETTINGS = {
    'initial_k': 20,  # Количество документов для первичного поиска
    'rerank_k': 8,    # Количество документов после переранжирования
    'hybrid_weight': 0.7,  # Вес семантического поиска в гибридном поиске
    'min_relevance': 0.5   # Минимальный порог релевантности
}

def clean_text(text: str) -> str:
    """Улучшенная очистка и нормализация текста"""
    # Удаляем лишние пробелы и переносы строк
    text = re.sub(r'\s+', ' ', text)
    # Удаляем специальные символы, сохраняя структуру
    text = re.sub(r'[^\w\s.,!?;:()\-–—\n\t]', '', text)
    # Нормализуем пробелы вокруг пунктуации
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # Восстанавливаем структуру параграфов
    text = re.sub(r'\.\s+', '.\n', text)
    return text.strip()

def extract_tables_from_pdf(file_path: str) -> List[Dict]:
    """Извлечение таблиц из PDF с улучшенной обработкой"""
    try:
        # Извлекаем таблицы с помощью tabula
        tables = tabula.read_pdf(
            file_path,
            **TABLE_SETTINGS
        )
        
        extracted_tables = []
        for i, table in enumerate(tables):
            if not table.empty:
                # Преобразуем таблицу в текст с сохранением структуры
                table_text = []
                for _, row in table.iterrows():
                    row_text = ' | '.join(str(cell).strip() for cell in row)
                    table_text.append(row_text)
                
                extracted_tables.append({
                    "text": f"Таблица {i+1}:\n" + "\n".join(table_text),
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "type": "table",
                        "table_index": i+1
                    }
                })
        
        return extracted_tables
    except Exception as e:
        logging.error(f"Ошибка извлечения таблиц из PDF: {str(e)}")
        return []

def extract_text_with_ocr(file_path: str) -> str:
    """Извлечение текста из PDF с использованием OCR"""
    try:
        # Конвертируем PDF в изображения
        images = convert_from_path(file_path)
        
        text_parts = []
        for i, image in enumerate(images):
            # Применяем OCR к каждому изображению
            text = pytesseract.image_to_string(
                image,
                **TESSERACT_CONFIG
            )
            text_parts.append(f"Страница {i+1}:\n{text}")
        
        return "\n\n".join(text_parts)
    except Exception as e:
        logging.error(f"Ошибка OCR: {str(e)}")
        return ""

class HybridSearchRetriever:
    """Реализация гибридного поиска, сочетающего семантический и ключевой поиск"""
    def __init__(self, vector_retriever, bm25_retriever, weights=None):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.weights = weights or {"vector": 0.7, "keyword": 0.3}
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[LangchainDocument]:
        return self.invoke(query, k=k)
    
    def invoke(self, query: str, k: int = 5) -> List[LangchainDocument]:
        # Получаем результаты от обоих поисковиков
        vector_docs = self.vector_retriever.get_relevant_documents(query, k=k*2)
        bm25_docs = self.bm25_retriever.get_relevant_documents(query, k=k*2)
        
        # Объединяем и ранжируем результаты
        seen_docs = set()
        scored_docs = []
        
        # Обрабатываем результаты векторного поиска
        for doc in vector_docs:
            if doc.page_content not in seen_docs:
                seen_docs.add(doc.page_content)
                scored_docs.append({
                    "document": doc,
                    "score": self.weights["vector"] * (1 - doc.metadata.get("score", 0))
                })
        
        # Обрабатываем результаты BM25
        for doc in bm25_docs:
            if doc.page_content not in seen_docs:
                seen_docs.add(doc.page_content)
                scored_docs.append({
                    "document": doc,
                    "score": self.weights["keyword"] * doc.metadata.get("score", 0)
                })
        
        # Сортируем по финальному скору
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        
        return [item["document"] for item in scored_docs[:k]]

class DocumentAnalyzer:
    def __init__(self, client, documents_dir: str = "documents"):
        self.logger = logging.getLogger('DocumentAnalyzer')
        self.client = client  # GigaChat client
        self.documents_dir = os.path.abspath(documents_dir)
        self.chroma_dir = CHROMA_DIR
        self.cache_dir = os.path.join(current_dir, '.cache')
        self.embeddings = None  # Инициализируется в _init_components
        self.llm = None  # Инициализируется в _init_components
        self.vector_store = None # Инициализируется в read_documents
        self.retrieval_chain = None # Инициализируется в _init_rag_chains
        self.generation_chain = None # Инициализируется в _init_rag_chains
        self.agent_orchestrator = None # Инициализируется в _init_agents
        self.bm25_retriever = None # Инициализируется в _create_retriever

        self.logger.info(f"Используется папка документов: {self.documents_dir}")
        self.logger.info(f"Используется папка Chroma: {self.chroma_dir}")

        # Убедимся, что директории существуют
        os.makedirs(self.documents_dir, exist_ok=True)

        # Удаление Chroma директории, если она существует и не используется
        # Это делается для обеспечения чистого состояния при каждом запуске для отладки
        # В продакшене, возможно, потребуется другая логика
        self._clean_chroma_dir()

        # Инициализация базовых компонентов (embeddings, llm)
        self._init_components()

        # Инициализация агентов (зависит от llm)
        self._init_agents()
        self.logger.info("Агенты инициализированы")

        # Чтение и индексация существующих документов
        self.read_documents()

        # Инициализация RAG-цепочек (зависит от vector_store)
        # Это должно произойти после того, как vector_store инициализирован в read_documents
        if self.vector_store:
             self._init_rag_chains()
             self.logger.info("RAG-цепи инициализированы")
        else:
             self.logger.warning("Векторное хранилище не инициализировано, RAG-цепи не будут инициализированы.")

        self.logger.info("Анализатор документов инициализирован")

    def _init_components(self):
        """Инициализация базовых компонентов"""
        try:
            # Создаем сплиттер для текста
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
            )
            
            # Инициализируем кэш
            self.cache = TTLCache(maxsize=MAX_CACHE_SIZE, ttl=CACHE_TTL)
            
            # Инициализируем эмбеддинги GigaChat
            credentials = base64.b64encode(
                f"{self.client.client_id}:{self.client.client_secret}".encode()
            ).decode()
            
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            self.embeddings = GigaChatEmbeddings(
                credentials=credentials,
                auth_url=self.client.auth_url,
                base_url=self.client.api_url,
                verify_ssl_certs=False,
                scope="GIGACHAT_API_PERS",
                model="EmbeddingsGigaR",  # Используем продвинутую модель
                timeout=30,
                ssl_context=ssl_context
            )
            
            # Инициализируем LLM
            self.llm = GigaChat(
                credentials=credentials,
                auth_url=self.client.auth_url,
                base_url=self.client.api_url,
                verify_ssl_certs=False,
                scope="GIGACHAT_API_PERS",
                model="GigaChat-Pro",  # Используем продвинутую модель
                timeout=30,
                ssl_context=ssl_context
            )
            
            self.logger.info("Компоненты успешно инициализированы")
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации компонентов: {str(e)}")
            raise

    def _clean_chroma_dir(self):
        """Очистка директории Chroma с учетом возможных блокировок"""
        try:
            if os.path.exists(self.chroma_dir):
                # Сначала пытаемся просто удалить содержимое директории
                for item in os.listdir(self.chroma_dir):
                    item_path = os.path.join(self.chroma_dir, item)
                    try:
                        if os.path.isfile(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path, ignore_errors=True)
                    except Exception as e:
                        self.logger.warning(f"Не удалось удалить {item_path}: {str(e)}")
                
                # Если директория пуста, удаляем её
                if not os.listdir(self.chroma_dir):
                    try:
                        os.rmdir(self.chroma_dir)
                        self.logger.info(f"Папка {self.chroma_dir} очищена")
                    except Exception as e:
                        self.logger.warning(f"Не удалось удалить пустую директорию {self.chroma_dir}: {str(e)}")
            else:
                self.logger.info(f"Папка {self.chroma_dir} не существует")
        except Exception as e:
            self.logger.error(f"Ошибка очистки chroma: {str(e)}")

    def _init_agents(self):
        """Инициализация специализированных агентов"""
        self.research_agent = Agent(
            role='Research Agent',
            goal='Анализ и извлечение ключевой информации из документов',
            backstory='Специалист по глубокому анализу документов и извлечению структурированной информации',
            verbose=True
        )
        
        self.writer_agent = Agent(
            role='Writer Agent',
            goal='Формирование структурированных ответов на основе извлеченной информации',
            backstory='Эксперт по составлению четких и информативных ответов',
            verbose=True
        )
        
        self.validator_agent = Agent(
            role='Validator Agent',
            goal='Проверка точности и релевантности ответов',
            backstory='Специалист по валидации и проверке качества ответов',
            verbose=True
        )
        
        self.crew = Crew(
            agents=[self.research_agent, self.writer_agent, self.validator_agent],
            process=Process.sequential
        )

    def _multi_level_processing(self, text: str, doc_type: str) -> List[Dict]:
        """Многоуровневая обработка документа"""
        levels = []
        
        # Уровень 1: Базовая структура
        structure = self._extract_document_structure(text, doc_type)
        levels.append({
            'level': 1,
            'type': 'structure',
            'content': structure
        })
        
        # Уровень 2: Семантические единицы
        semantic_units = self._extract_semantic_units(text)
        levels.append({
            'level': 2,
            'type': 'semantic',
            'content': semantic_units
        })
        
        # Уровень 3: Ключевые концепции
        concepts = self._extract_concepts(text)
        levels.append({
            'level': 3,
            'type': 'concepts',
            'content': concepts
        })
        
        return levels

    def _smart_chunking(self, docs: List[LangchainDocument], doc_type: str) -> List[LangchainDocument]:
        """Улучшенное разбиение на чанки с учетом структуры документа"""
        try:
            # Выбираем стратегию чанкинга
            strategy = CHUNK_STRATEGIES.get(doc_type, CHUNK_STRATEGIES['default'])
            
            # Создаем сплиттер с учетом типа документа
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=strategy['chunk_size'],
                chunk_overlap=strategy['chunk_overlap'],
                length_function=len,
                separators=strategy['separators'],
                is_separator_regex=False
            )
            
            # Разбиваем документы на чанки
            chunks = []
            for doc in docs:
                # Сохраняем оригинальные метаданные
                metadata = doc.metadata.copy()
                
                # Разбиваем текст на чанки
                doc_chunks = splitter.split_text(doc.page_content)
                
                # Создаем новые документы с обогащенными метаданными
                for i, chunk in enumerate(doc_chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": i,
                        "total_chunks": len(doc_chunks),
                        "chunk_size": len(chunk),
                        "position": i / len(doc_chunks)
                    })
                    
                    chunks.append(LangchainDocument(
                        page_content=chunk,
                        metadata=chunk_metadata
                    ))
            
            self.logger.info(f"Создано {len(chunks)} чанков из {len(docs)} документов")
            return chunks[:MAX_CHUNKS_PER_DOC]
            
        except Exception as e:
            self.logger.error(f"Ошибка при разбиении на чанки: {str(e)}")
            raise

    def hybrid_search(self, query: str, k: int = 5) -> List[Dict]:
        """Гибридный поиск с реранжировкой результатов"""
        if not self.vector_store:
            return []
            
        # Кэшируем результаты поиска
        cache_key = f"search_{hashlib.md5(query.encode()).hexdigest()}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        # Шаг 1: Семантический поиск
        semantic_results = self.vector_store.similarity_search_with_score(
            query,
            k=k*2  # Получаем больше результатов для реранжировки
        )
        
        # Шаг 2: Ключевое слово поиск
        keyword_results = self.vector_store.similarity_search_with_score(
            " ".join(query.split()[:3]),  # Используем первые 3 слова
            k=k*2
        )
        
        # Шаг 3: Объединяем и реранжируем результаты
        all_results = []
        seen_docs = set()
        
        for doc, score in semantic_results + keyword_results:
            if doc.page_content not in seen_docs:
                seen_docs.add(doc.page_content)
                all_results.append({
                    'document': doc,
                    'semantic_score': score,
                    'keyword_score': 0.0,
                    'final_score': 0.0
                })
        
        # Вычисляем финальные оценки
        for result in all_results:
            # Нормализуем оценки
            semantic_score = 1 / (1 + result['semantic_score'])
            keyword_score = 1 / (1 + result['keyword_score'])
            
            # Взвешенная сумма
            result['final_score'] = 0.7 * semantic_score + 0.3 * keyword_score
        
        # Сортируем по финальной оценке
        all_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Кэшируем результаты
        final_results = all_results[:k]
        self.search_cache[cache_key] = final_results
        
        return final_results

    def analyze_document(self, query: str, doc_path: Optional[str] = None) -> Dict:
        """Анализ документа с использованием многоагентной системы"""
        try:
            # Подготовка контекста
            if doc_path:
                context = self._get_document_context(doc_path)
            else:
                # Используем гибридный поиск для получения релевантных фрагментов
                search_results = self.hybrid_search(query)
                context = "\n".join([r['document'].page_content for r in search_results])
            
            # Создаем задачи для агентов
            research_task = Task(
                description=f"Проанализируй следующий контекст и извлеки ключевую информацию по запросу: {query}\n\nКонтекст:\n{context}",
                agent=self.research_agent
            )
            
            writer_task = Task(
                description="На основе извлеченной информации составь структурированный ответ",
                agent=self.writer_agent
            )
            
            validator_task = Task(
                description="Проверь точность и релевантность ответа",
                agent=self.validator_agent
            )
            
            # Запускаем выполнение задач
            result = self.crew.kickoff()
            
            return {
                'status': 'success',
                'answer': result,
                'sources': [r['document'].metadata for r in search_results] if not doc_path else None
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка при анализе документа: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def read_documents(self):
        """Чтение и индексация документов из папки"""
        if not os.path.exists(self.documents_dir):
            self.logger.warning(f"Папка документов {self.documents_dir} не найдена.")
            return

        self.logger.info(f"Начало индексации из {self.documents_dir}")
        documents = self._load_all_documents() # Загружаем все документы

        if documents:
            # Создаем векторное хранилище с загруженными документами
            try:
                self.vector_store = self._create_vector_store(documents)
                self.logger.info(f"Успешно проиндексировано {len(documents)} секций")
            except Exception as e:
                 self.logger.error(f"Ошибка индексации: {str(e)}")
        else:
            self.logger.info("В папке документов нет файлов для индексации.")

        self.logger.info("Индексация завершена")

        # После успешной загрузки документов и создания хранилища, инициализируем RAG-цепочки, если они еще не были инициализированы
        if self.vector_store and not self.retrieval_chain:
             self._init_rag_chains()
             self.logger.info("RAG-цепи инициализированы после загрузки документов")

    def _load_all_documents(self) -> List[LangchainDocument]:
        """Улучшенная загрузка документов с учетом структуры"""
        all_sections = []
        
        for filename in os.listdir(self.documents_dir):
            file_path = os.path.join(self.documents_dir, filename)
            doc_type = os.path.splitext(filename)[1].lower().lstrip('.')
            
            try:
                # Извлекаем структуру документа
                structure = self.extract_document_structure(file_path, doc_type)
                
                # Обрабатываем секции
                for section in structure["sections"]:
                    if section["type"] == "text":
                        chunks = self._smart_chunking(
                            [LangchainDocument(
                                page_content=section["content"],
                                metadata={
                                    "source": filename,
                                    "section": "text",
                                    "page": section.get("page"),
                                    "structure": "text"
                                }
                            )],
                            doc_type
                        )
                        all_sections.extend(chunks)
                    elif section["type"] == "heading":
                        # Обрабатываем заголовки и связанные параграфы
                        heading_text = section["content"]
                        paragraphs_text = "\n".join(section["paragraphs"])
                        chunks = self._smart_chunking(
                            [LangchainDocument(
                                page_content=f"{heading_text}\n{paragraphs_text}",
                                metadata={
                                    "source": filename,
                                    "section": f"heading_{section['level']}",
                                    "structure": "heading"
                                }
                            )],
                            doc_type
                        )
                        all_sections.extend(chunks)
                    elif section["type"] == "sheet":
                        # Обрабатываем листы Excel
                        content = json.dumps(section["content"], ensure_ascii=False)
                        chunks = self._smart_chunking(
                            [LangchainDocument(
                                page_content=content,
                                metadata={
                                    "source": filename,
                                    "section": f"sheet_{section['name']}",
                                    "structure": "table"
                                }
                            )],
                            doc_type
                        )
                        all_sections.extend(chunks)
                
                # Обрабатываем таблицы
                for table in structure["tables"]:
                    table_text = json.dumps(table["content"], ensure_ascii=False)
                    chunks = self._smart_chunking(
                        [LangchainDocument(
                            page_content=table_text,
                            metadata={
                                "source": filename,
                                "section": f"table_{table['table_index']}",
                                "page": table.get("page"),
                                "structure": "table"
                            }
                        )],
                        doc_type
                    )
                    all_sections.extend(chunks)
                
                self.logger.info(f"Обработан: {filename} ({len(structure['sections'])} секций, {len(structure['tables'])} таблиц)")
                
            except Exception as e:
                self.logger.error(f"Ошибка обработки файла {filename}: {str(e)}")
                continue
        
        return all_sections

    def _create_vector_store(self, documents: List[LangchainDocument]) -> Chroma:
        """Создание или загрузка векторного хранилища Chroma"""
        try:
            # Всегда создаем новое хранилище с переданными документами
            # Удаляем старое, если существует
            if os.path.exists(self.chroma_dir):
                try:
                    shutil.rmtree(self.chroma_dir)
                    self.logger.info(f"Папка {self.chroma_dir} очищена для создания нового хранилища")
                except PermissionError:
                    self.logger.warning(f"Не удалось удалить {self.chroma_dir}. Возможно, используется другим процессом. Попробуем создать новое хранилище в другом месте или использовать существующее с осторожностью.")

            # Создаем новую папку если её нет
            os.makedirs(self.chroma_dir, exist_ok=True)

            # Создаем клиент Chroma с новыми настройками
            client = PersistentClient(path=self.chroma_dir)

            # Создаем хранилище с новым клиентом
            vector_store = Chroma(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings
            )

            # Добавляем документы в новое хранилище, если они есть
            if documents:
                vector_store.add_documents(documents)
                self.logger.info(f"Добавлено {len(documents)} документов в новое хранилище")

            return vector_store

        except Exception as e:
            self.logger.error(f"Ошибка создания хранилища: {str(e)}")
            # Важно поднять исключение, чтобы предотвратить дальнейшую работу с некорректным хранилищем
            raise

    def _create_retriever(self):
        """Создание улучшенного поисковика с гибридным поиском"""
        try:
            # Базовая конфигурация векторного поисковика
            vector_retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": SEARCH_SETTINGS['initial_k']
                }
            )

            # Получаем документы из Chroma и преобразуем их в формат LangchainDocument
            docs = []
            chroma_docs = self.vector_store.get()
            
            # Проверяем, что у нас есть документы
            if not chroma_docs or not chroma_docs.get('documents'):
                self.logger.warning("В хранилище нет документов для создания BM25 поисковика")
                return vector_retriever

            # Преобразуем документы в формат LangchainDocument
            for i, doc_text in enumerate(chroma_docs['documents']):
                metadata = chroma_docs.get('metadatas', [{}])[i] if chroma_docs.get('metadatas') else {}
                docs.append(LangchainDocument(
                    page_content=doc_text,
                    metadata=metadata
                ))

            # Создаем BM25 поисковик для ключевых слов
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            
            # Создаем гибридный поисковик
            hybrid_retriever = HybridSearchRetriever(
                vector_retriever=vector_retriever,
                bm25_retriever=self.bm25_retriever,
                weights={
                    "vector": SEARCH_SETTINGS['hybrid_weight'],
                    "keyword": 1 - SEARCH_SETTINGS['hybrid_weight']
                }
            )
            
            self.logger.info("Поисковик успешно создан")
            return hybrid_retriever
            
        except Exception as e:
            self.logger.error(f"Ошибка создания поисковика: {str(e)}")
            return vector_retriever

    def add_document(self, file_path: str) -> bool:
        """Добавление нового документа и обновление индекса"""
        try:
            target_path = os.path.join(self.documents_dir, os.path.basename(file_path))

            # Проверяем, существует ли файл уже в директории документов
            if os.path.exists(target_path):
                self.logger.warning(f"Файл {os.path.basename(file_path)} уже существует в папке документов.")
                return False

            # Копируем файл в папку документов
            shutil.copy(file_path, target_path)
            self.logger.info(f"Файл {os.path.basename(file_path)} скопирован в {self.documents_dir}")

            # Загружаем документ
            doc_result = self._load_document(target_path)
            if not doc_result or not doc_result.get("documents"):
                self.logger.warning(f"Не удалось загрузить документ {os.path.basename(file_path)}")
                try:
                    os.remove(target_path)
                except Exception as remove_e:
                    self.logger.error(f"Ошибка удаления файла {os.path.basename(file_path)}: {str(remove_e)}")
                return False

            # Добавляем документы в хранилище
            if self.vector_store:
                self.vector_store.add_documents(doc_result["documents"])
                self.logger.info(f"Добавлено {len(doc_result['documents'])} секций из файла {os.path.basename(file_path)} в индекс")
                
                # Обновляем BM25 индекс
                if self.bm25_retriever:
                    self.bm25_retriever.add_documents(doc_result["documents"])
            else:
                # Если хранилище еще не создано, создаем его
                self.vector_store = self._create_vector_store(doc_result["documents"])
                self.logger.info(f"Создано новое хранилище с {len(doc_result['documents'])} секциями")
                self._init_rag_chains()

            return True

        except Exception as e:
            self.logger.error(f"Ошибка добавления документа: {str(e)}")
            return False

    def _init_rag_chains(self):
        """Инициализация цепочек RAG для индексации и поиска"""
        try:
            # Создаем поисковик
            retriever = self._create_retriever()
            
            # Создаем промпт для генерации
            prompt = self._create_prompt_template()
            
            # Инициализируем цепочки
            self.retrieval_chain = {
                "retriever": retriever,
                "reranker": self._create_reranker(),
                "context_builder": self._build_context
            }
            
            self.generation_chain = {
                "prompt": prompt,
                "llm": self.llm,
                "post_processor": self._post_process_response
            }
            
            self.logger.info("RAG-цепи инициализированы")
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации RAG-цепей: {str(e)}")
            raise

    def _create_reranker(self):
        """Создание компонента для реранжировки результатов"""
        return {
            "similarity": lambda x, y: cosine_similarity(
                [self.embeddings.embed_query(x)],
                [self.embeddings.embed_query(y)]
            )[0][0],
            "keyword_match": lambda x, y: self._calculate_keyword_overlap(x, y),
            "position_score": lambda doc: 1 - doc.metadata.get('position', 0.5),
            "structure_score": lambda doc: self._calculate_structure_score(doc)
        }

    def _build_context(self, query: str, retrieved_docs: List[LangchainDocument]) -> str:
        """Построение контекста из найденных документов"""
        try:
            # Группируем документы по источникам
            source_groups = {}
            for doc in retrieved_docs:
                source = doc.metadata.get('source', 'unknown')
                if source not in source_groups:
                    source_groups[source] = []
                source_groups[source].append(doc)
            
            # Строим контекст с учетом структуры
            context_parts = []
            for source, docs in source_groups.items():
                # Сортируем документы по позиции
                sorted_docs = sorted(docs, key=lambda x: x.metadata.get('position', 0))
                
                # Добавляем заголовок источника
                context_parts.append(f"Источник: {source}")
                
                # Обрабатываем документы с учетом их типа
                for doc in sorted_docs:
                    structure_type = doc.metadata.get('structure', 'text')
                    if structure_type == 'table':
                        context_parts.append(f"Таблица:\n{doc.page_content}")
                    elif structure_type == 'heading':
                        context_parts.append(f"Заголовок и текст:\n{doc.page_content}")
                    else:
                        context_parts.append(doc.page_content)
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            self.logger.error(f"Ошибка построения контекста: {str(e)}")
            return "\n".join(doc.page_content for doc in retrieved_docs)

    def _create_prompt_template(self) -> PromptTemplate:
        """Создание шаблона промпта для генерации ответов"""
        template = """Ты - эксперт по анализу документов. Используй предоставленный контекст для ответа на вопрос.
        
Контекст:
{context}

Вопрос: {question}

Инструкции:
1. Используй ТОЛЬКО информацию из контекста
2. Если информация противоречива, укажи это
3. Структурируй ответ с помощью:
   - Маркированных списков
   - Подзаголовков
   - Цитат из контекста
4. Укажи источники информации
5. Если информации недостаточно, признай это
6. НЕ добавляй информацию, которой нет в контексте

Ответ:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def _post_process_response(self, response: str, sources: List[Dict]) -> Dict:
        """Пост-обработка ответа"""
        try:
            # Извлекаем источники из ответа
            source_pattern = r"Источники?:?\s*\[(.*?)\]"
            sources_match = re.search(source_pattern, response, re.IGNORECASE)
            
            # Форматируем ответ
            formatted_response = {
                "answer": response,
                "sources": sources,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "processing_time": time.time() - self._start_time,
                    "sources_count": len(sources)
                }
            }
            
            return formatted_response
            
        except Exception as e:
            self.logger.error(f"Ошибка пост-обработки ответа: {str(e)}")
            return {"answer": response, "sources": sources}

    def analyze_documents(self, query: str) -> Dict:
        """Улучшенный анализ документов с оптимизированным RAG-конвейером"""
        try:
            self._start_time = time.time()
            
            # Проверяем кэш
            query_hash = hashlib.md5(query.encode()).hexdigest()
            cached_response = self._get_cached_response(query_hash)
            if cached_response:
                return cached_response
            
            # Шаг 1: Поиск релевантных документов
            retrieved_docs = self.retrieval_chain["retriever"].invoke(query)
            
            # Шаг 2: Реранжировка результатов
            reranked_docs = self._rerank_documents(query, retrieved_docs)
            
            # Шаг 3: Построение контекста
            context = self.retrieval_chain["context_builder"](query, reranked_docs)
            
            # Шаг 4: Генерация ответа
            prompt = self.generation_chain["prompt"].format(
                context=context,
                question=query
            )
            
            response = self.generation_chain["llm"].predict(prompt)
            
            # Шаг 5: Пост-обработка
            final_response = self.generation_chain["post_processor"](
                response,
                [{"source": doc.metadata.get("source"), 
                  "relevance": doc.metadata.get("score", 0)} 
                 for doc in reranked_docs]
            )
            
            # Сохраняем в кэш
            self._save_to_cache(query_hash, final_response)
            
            return final_response
            
        except Exception as e:
            self.logger.error(f"Ошибка в RAG-конвейере: {str(e)}")
            return {"error": str(e)}

    def _rerank_documents(self, query: str, docs: List[LangchainDocument]) -> List[LangchainDocument]:
        """Реранжировка документов с учетом релевантности"""
        try:
            scored_docs = []
            for doc in docs:
                # Семантическая релевантность
                semantic_score = self.retrieval_chain["reranker"]["similarity"](query, doc.page_content)
                
                # Перекрытие ключевых слов
                keyword_score = self.retrieval_chain["reranker"]["keyword_match"](query, doc.page_content)
                
                # Оценка структуры
                structure_score = self.retrieval_chain["reranker"]["structure_score"](doc)
                
                # Позиционная оценка
                position_score = self.retrieval_chain["reranker"]["position_score"](doc)
                
                # Вычисляем финальный скор
                final_score = (
                    0.4 * semantic_score +
                    0.3 * keyword_score +
                    0.2 * structure_score +
                    0.1 * position_score
                )
                
                # Обновляем метаданные
                doc.metadata["score"] = final_score
                scored_docs.append(doc)
            
            # Сортируем по финальному скору
            scored_docs.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)
            
            return scored_docs[:SEARCH_SETTINGS['rerank_k']]
            
        except Exception as e:
            self.logger.error(f"Ошибка реранжировки документов: {str(e)}")
            return docs

    def _load_document(self, file_path: str) -> Dict:
        """Улучшенная загрузка документов с использованием Unstructured loaders"""
        try:
            file_type = os.path.splitext(file_path)[1].lower().lstrip('.')
            self.logger.info(f"Загрузка документа: {file_path} (тип: {file_type})")
            
            # Выбираем подходящий загрузчик
            if file_type == 'pdf':
                loader = UnstructuredPDFLoader(
                    file_path,
                    mode="single",  # или "elements" для более детальной структуры
                    strategy="fast"  # или "accurate" для лучшего качества
                )
            elif file_type in ['docx', 'doc']:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_type in ['xlsx', 'xls']:
                loader = UnstructuredExcelLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding='utf-8')
            
            # Загружаем документ
            docs = loader.load()
            
            # Извлекаем структуру
            structure = self.extract_document_structure(file_path, file_type)
            
            # Обогащаем метаданные
            for doc in docs:
                doc.metadata.update({
                    "source": os.path.basename(file_path),
                    "type": file_type,
                    "timestamp": datetime.now().isoformat(),
                    "structure": structure
                })
            
            return {
                "documents": docs,
                "structure": structure,
                "metadata": {
                    "source": os.path.basename(file_path),
                    "type": file_type,
                    "timestamp": datetime.now().isoformat(),
                    "pages_count": len(docs)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки документа {file_path}: {str(e)}")
            raise

    def _update_index(self, new_docs: List[LangchainDocument]) -> None:
        """Инкрементальное обновление индекса"""
        try:
            if not self.vector_store:
                self.vector_store = self._create_vector_store(new_docs)
                return
            
            # Добавляем новые документы в существующий индекс
            self.vector_store.add_documents(new_docs)
            
            # Обновляем BM25 индекс
            if self.bm25_retriever:
                texts = [doc.page_content for doc in new_docs]
                self.bm25_retriever.add_documents(
                    [LangchainDocument(page_content=text) for text in texts]
                )
            
            self.logger.info(f"Индекс обновлен: добавлено {len(new_docs)} документов")
            
        except Exception as e:
            self.logger.error(f"Ошибка обновления индекса: {str(e)}")
            raise

    def ask_question(self, question: str, doc_path: Optional[str] = None) -> str:
        """Задавание вопроса по документу или всем документам"""
        try:
            # Используем существующий метод analyze_documents
            result = self.analyze_documents(question)
            
            if "error" in result:
                raise Exception(result["error"])
                
            return result["answer"]
            
        except Exception as e:
            self.logger.error(f"Ошибка при задавании вопроса: {str(e)}")
            return f"Извините, произошла ошибка при обработке вопроса: {str(e)}"

    def extract_document_structure(self, file_path: str, doc_type: str) -> Dict:
        """Извлечение структуры документа с учетом его типа"""
        structure = {
            "sections": [],
            "tables": [],
            "images": [],
            "metadata": {}
        }
        
        try:
            if doc_type == 'pdf':
                # Используем PyPDF2 для извлечения структуры
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    structure["metadata"].update({
                        "pages": len(pdf.pages),
                        "title": pdf.metadata.get('/Title', ''),
                        "author": pdf.metadata.get('/Author', '')
                    })
                    
                    # Анализируем структуру каждой страницы
                    for i, page in enumerate(pdf.pages):
                        # Извлекаем текст с сохранением позиции
                        text = page.extract_text()
                        if text.strip():
                            structure["sections"].append({
                                "type": "text",
                                "page": i + 1,
                                "content": text
                            })
                        
                        # Ищем таблицы
                        tables = tabula.read_pdf(file_path, pages=i+1)
                        for j, table in enumerate(tables):
                            if not table.empty:
                                structure["tables"].append({
                                    "page": i + 1,
                                    "table_index": j + 1,
                                    "content": table.to_dict('records')
                                })
            
            elif doc_type in ['docx', 'doc']:
                doc = Document(file_path)
                
                # Извлекаем метаданные
                core_props = doc.core_properties
                structure["metadata"].update({
                    "title": core_props.title,
                    "author": core_props.author,
                    "created": core_props.created,
                    "modified": core_props.modified
                })
                
                # Анализируем структуру документа
                current_section = None
                for para in doc.paragraphs:
                    if para.style.name.startswith('Heading'):
                        if current_section:
                            structure["sections"].append(current_section)
                        current_section = {
                            "type": "heading",
                            "level": int(para.style.name[-1]),
                            "content": para.text,
                            "paragraphs": []
                        }
                    elif current_section:
                        current_section["paragraphs"].append(para.text)
                    else:
                        structure["sections"].append({
                            "type": "text",
                            "content": para.text
                        })
                
                # Добавляем последнюю секцию
                if current_section:
                    structure["sections"].append(current_section)
                
                # Извлекаем таблицы
                for i, table in enumerate(doc.tables):
                    table_data = []
                    for row in table.rows:
                        table_data.append([cell.text for cell in row.cells])
                    structure["tables"].append({
                        "table_index": i + 1,
                        "content": table_data
                    })
            
            elif doc_type in ['xlsx', 'xls']:
                excel = pd.ExcelFile(file_path)
                structure["metadata"]["sheets"] = excel.sheet_names
                
                for sheet_name in excel.sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    structure["sections"].append({
                        "type": "sheet",
                        "name": sheet_name,
                        "content": df.to_dict('records')
                    })
            
            # Добавляем метаданные файла
            stat = os.stat(file_path)
            structure["metadata"].update({
                "name": os.path.basename(file_path),
                "type": doc_type,
                "size": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime
            })
            
            return structure
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения структуры документа: {str(e)}")
            return {
                "sections": [],
                "tables": [],
                "images": [],
                "metadata": {
                    "name": os.path.basename(file_path),
                    "type": doc_type,
                    "error": str(e)
                }
            }

    def _get_cached_response(self, query_hash: str) -> Optional[Dict]:
        """Получение ответа из кэша"""
        try:
            return self.cache.get(query_hash)
        except Exception as e:
            self.logger.warning(f"Ошибка получения из кэша: {str(e)}")
            return None

    def _save_to_cache(self, query_hash: str, response: Dict) -> None:
        """Сохранение ответа в кэш"""
        try:
            self.cache[query_hash] = response
        except Exception as e:
            self.logger.warning(f"Ошибка сохранения в кэш: {str(e)}")

    def _extract_concepts(self, text: str) -> List[Dict]:
        """Извлечение ключевых концептов из текста для построения графа"""
        try:
            # Используем GigaChat для извлечения концептов
            prompt = f"""Извлеки ключевые концепты из следующего текста. 
            Для каждого концепта укажи:
            1. Название
            2. Описание
            3. Связанные концепты
            4. Важность (от 0 до 1)
            
            Текст:
            {text}
            
            Ответ должен быть в формате JSON:
            {{
                "concepts": [
                    {{
                        "name": "название концепта",
                        "description": "описание",
                        "related": ["связанный концепт 1", "связанный концепт 2"],
                        "importance": 0.8
                    }}
                ]
            }}
            """
            
            response = self.client.chat(prompt)
            concepts_data = json.loads(response.choices[0].message.content)
            
            # Создаем граф концептов
            concepts = []
            for concept in concepts_data.get("concepts", []):
                concepts.append({
                    "id": hashlib.md5(concept["name"].encode()).hexdigest(),
                    "name": concept["name"],
                    "description": concept["description"],
                    "importance": concept["importance"],
                    "related": concept["related"]
                })
            
            return concepts
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения концептов: {str(e)}")
            return []

    def _extract_semantic_units(self, text: str) -> List[Dict]:
        """Извлечение семантических единиц из текста"""
        try:
            # Разбиваем текст на предложения
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            units = []
            for i, sentence in enumerate(sentences):
                # Определяем тип единицы
                unit_type = "statement"
                if sentence.endswith("?"):
                    unit_type = "question"
                elif sentence.endswith("!"):
                    unit_type = "exclamation"
                
                # Извлекаем ключевые слова
                words = re.findall(r'\w+', sentence.lower())
                keywords = [w for w in words if len(w) > 3]  # Игнорируем короткие слова
                
                units.append({
                    "id": f"unit_{i}",
                    "type": unit_type,
                    "content": sentence,
                    "keywords": keywords,
                    "position": i / len(sentences)
                })
            
            return units
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения семантических единиц: {str(e)}")
            return []

    def _calculate_structure_score(self, doc: LangchainDocument) -> float:
        """Расчет оценки структуры документа"""
        try:
            # Базовые веса для разных типов структур
            structure_weights = {
                "heading": 1.0,
                "table": 0.8,
                "text": 0.6
            }
            
            # Получаем тип структуры
            structure_type = doc.metadata.get("structure", "text")
            
            # Базовая оценка на основе типа
            base_score = structure_weights.get(structure_type, 0.5)
            
            # Корректируем оценку на основе позиции
            position = doc.metadata.get("position", 0.5)
            position_factor = 1 - abs(position - 0.5) * 0.5  # Предпочитаем середину документа
            
            # Корректируем на основе размера чанка
            chunk_size = doc.metadata.get("chunk_size", 0)
            size_factor = min(chunk_size / 1000, 1.0)  # Нормализуем размер
            
            return base_score * position_factor * (0.7 + 0.3 * size_factor)
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета оценки структуры: {str(e)}")
            return 0.5

    def _calculate_keyword_overlap(self, query: str, text: str) -> float:
        """Расчет перекрытия ключевых слов"""
        try:
            # Извлекаем ключевые слова из запроса и текста
            query_words = set(re.findall(r'\w+', query.lower()))
            text_words = set(re.findall(r'\w+', text.lower()))
            
            # Игнорируем короткие слова
            query_words = {w for w in query_words if len(w) > 3}
            text_words = {w for w in text_words if len(w) > 3}
            
            if not query_words or not text_words:
                return 0.0
            
            # Считаем пересечение
            overlap = len(query_words & text_words)
            
            # Нормализуем результат
            return overlap / len(query_words)
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета перекрытия ключевых слов: {str(e)}")
            return 0.0