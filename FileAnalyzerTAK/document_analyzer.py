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
from langchain_community.vectorstores import Chroma
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

def extract_document_structure(file_path: str, file_type: str) -> Dict:
    """Извлечение структуры документа с учетом его типа"""
    structure = {
        "sections": [],
        "tables": [],
        "images": [],
        "metadata": {}
    }
    
    try:
        if file_type == 'pdf':
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
        
        elif file_type in ['docx', 'doc']:
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
        
        elif file_type in ['xlsx', 'xls']:
            excel = pd.ExcelFile(file_path)
            structure["metadata"]["sheets"] = excel.sheet_names
            
            for sheet_name in excel.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                structure["sections"].append({
                    "type": "sheet",
                    "name": sheet_name,
                    "content": df.to_dict('records')
                })
        
        return structure
        
    except Exception as e:
        logging.error(f"Ошибка извлечения структуры документа: {str(e)}")
        return structure

def smart_chunking(text: str, metadata: Dict, doc_type: str) -> List[Dict]:
    """Улучшенное разбиение текста на чанки с учетом структуры и типа документа"""
    # Выбираем стратегию чанкинга в зависимости от типа документа
    strategy = CHUNK_STRATEGIES.get(doc_type, CHUNK_STRATEGIES['default'])
    
    # Создаем сплиттер с учетом стратегии
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=strategy['chunk_size'],
        chunk_overlap=strategy['chunk_overlap'],
        length_function=len,
        separators=strategy['separators']
    )
    
    # Разбиваем текст на чанки
    chunks = splitter.split_text(text)
    
    # Добавляем метаданные и информацию о структуре
    return [
        {
            "text": chunk,
            "metadata": {
                **metadata,
                "chunk_index": i,
                "doc_type": doc_type,
                "chunk_size": len(chunk),
                "position": i / len(chunks)  # Относительная позиция в документе
            }
        }
        for i, chunk in enumerate(chunks)
    ]

class GigaChatEmbedder:
    def __init__(self, client):
        credentials = base64.b64encode(
            f"{client.client_id}:{client.client_secret}".encode()
        ).decode()
        
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Используем модель EmbeddingsGigaR для лучшей точности
        self.embeddings = GigaChatEmbeddings(
            credentials=credentials,
            auth_url=client.auth_url,
            base_url=client.api_url,
            verify_ssl_certs=False,
            scope="GIGACHAT_API_PERS",
            model="EmbeddingsGigaR",  # Используем продвинутую модель
            timeout=30,
            ssl_context=ssl_context
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Получение эмбеддингов для документов"""
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Получение эмбеддинга для запроса с добавлением инструкции"""
        # Добавляем инструкцию для улучшения точности поиска
        instruction = "Дан вопрос, необходимо найти абзац текста с ответом\nвопрос: "
        return self.embeddings.embed_query(f"{instruction}{text}")
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)

class HybridSearchRetriever:
    """Реализация гибридного поиска, сочетающего семантический и ключевой поиск"""
    def __init__(self, vector_retriever, bm25_retriever, weights=None):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.weights = weights or {"vector": 0.7, "keyword": 0.3}
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
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
        self.client = client
        self.documents_dir = documents_dir
        self.logger = logging.getLogger('DocumentAnalyzer')
        
        # Инициализация кэшей
        self.text_cache = TTLCache(maxsize=MAX_CACHE_SIZE, ttl=CACHE_TTL)
        self.embedding_cache = TTLCache(maxsize=MAX_CACHE_SIZE, ttl=CACHE_TTL)
        self.search_cache = TTLCache(maxsize=MAX_CACHE_SIZE, ttl=CACHE_TTL)
        
        # Инициализация эмбеддингов и векторного хранилища
        self.embeddings = GigaChatEmbedder(client)
        self.vector_store = None
        self.bm25_retriever = None
        
        # Инициализация цепочек RAG
        self.retrieval_chain = None
        self.generation_chain = None
        
        # Создание директории для документов
        os.makedirs(documents_dir, exist_ok=True)
        
        self.logger.info(f"Используется папка документов: {self.documents_dir}")
        self.logger.info(f"Используется папка Chroma: {CHROMA_DIR}")
        
        self._clean_chroma_dir()
        self.cache_dir = os.path.join(current_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._init_components()
        self._init_rag_chains()
        self._init_agents()
        
        self.logger.info("Анализатор документов инициализирован")

    def _clean_chroma_dir(self):
        try:
            if os.path.exists(CHROMA_DIR):
                shutil.rmtree(CHROMA_DIR, ignore_errors=True)
                self.logger.info(f"Папка {CHROMA_DIR} очищена")
            else:
                self.logger.info(f"Папка {CHROMA_DIR} не существует")
        except Exception as e:
            self.logger.error(f"Ошибка очистки chroma: {str(e)}")

    def _init_components(self):
        try:
            self.embedder = GigaChatEmbedder(self.client)
            
            credentials = base64.b64encode(
                f"{self.client.client_id}:{self.client.client_secret}".encode()
            ).decode()
            
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            self.llm = GigaChat(
                credentials=credentials,
                auth_url=self.client.auth_url,
                base_url=self.client.api_url,
                verify_ssl_certs=False,
                scope="GIGACHAT_API_PERS",
                temperature=0.7,
                max_tokens=1500,
                ssl_context=ssl_context
            )
            
            self.logger.info("Компоненты LangChain инициализированы")
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации: {str(e)}")
            raise

    @lru_cache(maxsize=100)
    def _get_cached_response(self, query_hash: str) -> Optional[Dict]:
        """Получение кэшированного ответа"""
        cache_file = os.path.join(self.cache_dir, f"{query_hash}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    if datetime.fromisoformat(cached_data['timestamp']) > datetime.now() - timedelta(days=1):
                        return cached_data['response']
            except Exception as e:
                self.logger.warning(f"Ошибка чтения кэша: {str(e)}")
        return None

    def _save_to_cache(self, query_hash: str, response: Dict):
        """Сохранение ответа в кэш"""
        cache_file = os.path.join(self.cache_dir, f"{query_hash}.json")
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'response': response
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Ошибка сохранения в кэш: {str(e)}")

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

    def _smart_chunking(self, text: str, doc_type: str) -> List[str]:
        """Умное разделение на чанки с учетом структуры документа"""
        chunks = []
        
        # Определяем стратегию чанкинга в зависимости от типа документа
        if doc_type == 'pdf':
            # Для PDF используем разделение по страницам и параграфам
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
            )
        elif doc_type in ['docx', 'doc']:
            # Для Word используем разделение по разделам и параграфам
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
            )
        elif doc_type in ['xlsx', 'xls']:
            # Для Excel используем разделение по таблицам
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n", "\t", " ", ""]
            )
        else:
            # Для остальных типов используем стандартное разделение
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
        
        # Разбиваем текст на чанки
        raw_chunks = splitter.split_text(text)
        
        # Обрабатываем каждый чанк
        for chunk in raw_chunks:
            # Добавляем метаданные к чанку
            chunk_metadata = {
                'type': doc_type,
                'timestamp': datetime.now().isoformat(),
                'chunk_size': len(chunk)
            }
            
            # Создаем документ с метаданными
            doc = Document(
                page_content=chunk,
                metadata=chunk_metadata
            )
            
            chunks.append(doc)
        
        return chunks[:MAX_CHUNKS_PER_DOC]

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
        try:
            self.logger.info(f"Начало индексации из {self.documents_dir}")
            
            if not os.path.exists(self.documents_dir):
                os.makedirs(self.documents_dir)
                self.logger.warning(f"Создана папка документов: {self.documents_dir}")
                return
                
            documents = self._load_all_documents()
            if documents:
                self.vector_store = self._create_vector_store(documents)
                self.rag_chain = self._create_rag_chain()
                # Инициализируем оркестратор агентов
                from agents import AgentOrchestrator
                self.agent_orchestrator = AgentOrchestrator(self.llm, self.vector_store)
                self.logger.info(f"Успешно проиндексировано {len(documents)} секций")
            else:
                self.logger.warning("Нет документов для индексации")
                
        except Exception as e:
            self.logger.error(f"Ошибка индексации: {str(e)}")
            raise

    def _load_all_documents(self) -> List[LangchainDocument]:
        """Улучшенная загрузка документов с учетом структуры"""
        all_sections = []
        
        for filename in os.listdir(self.documents_dir):
            file_path = os.path.join(self.documents_dir, filename)
            doc_type = os.path.splitext(filename)[1].lower().lstrip('.')
            
            try:
                # Извлекаем структуру документа
                structure = extract_document_structure(file_path, doc_type)
                
                # Обрабатываем секции
                for section in structure["sections"]:
                    if section["type"] == "text":
                        chunks = smart_chunking(
                            section["content"],
                            {
                                "source": filename,
                                "section": "text",
                                "page": section.get("page"),
                                "structure": "text"
                            },
                            doc_type
                        )
                        all_sections.extend(chunks)
                    elif section["type"] == "heading":
                        # Обрабатываем заголовки и связанные параграфы
                        heading_text = section["content"]
                        paragraphs_text = "\n".join(section["paragraphs"])
                        chunks = smart_chunking(
                            f"{heading_text}\n{paragraphs_text}",
                            {
                                "source": filename,
                                "section": f"heading_{section['level']}",
                                "structure": "heading"
                            },
                            doc_type
                        )
                        all_sections.extend(chunks)
                    elif section["type"] == "sheet":
                        # Обрабатываем листы Excel
                        content = json.dumps(section["content"], ensure_ascii=False)
                        chunks = smart_chunking(
                            content,
                            {
                                "source": filename,
                                "section": f"sheet_{section['name']}",
                                "structure": "table"
                            },
                            doc_type
                        )
                        all_sections.extend(chunks)
                
                # Обрабатываем таблицы
                for table in structure["tables"]:
                    table_text = json.dumps(table["content"], ensure_ascii=False)
                    chunks = smart_chunking(
                        table_text,
                        {
                            "source": filename,
                            "section": f"table_{table['table_index']}",
                            "page": table.get("page"),
                            "structure": "table"
                        },
                        doc_type
                    )
                    all_sections.extend(chunks)
                
                self.logger.info(f"Обработан: {filename} ({len(structure['sections'])} секций, {len(structure['tables'])} таблиц)")
                
            except Exception as e:
                self.logger.error(f"Ошибка обработки файла {filename}: {str(e)}")
                continue
        
        # Преобразуем секции в документы LangChain
        documents = [
            LangchainDocument(
                page_content=section["text"],
                metadata=section["metadata"]
            ) for section in all_sections
        ]
        
        return documents

    def _create_vector_store(self, documents: List[LangchainDocument]) -> Chroma:
        """Создание векторного хранилища с гибридным поиском"""
        try:
            self._clean_chroma_dir()
            
            # Настройка Chroma
            chroma_settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
            
            # Создаем TF-IDF векторизатор для полнотекстового поиска
            texts = [doc.page_content for doc in documents]
            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform(texts)
            
            # Создаем векторное хранилище
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedder.embeddings,
                persist_directory=CHROMA_DIR,
                collection_name=COLLECTION_NAME,
                client_settings=chroma_settings
            )
            
            # Сохраняем TF-IDF матрицу для гибридного поиска
            self.tfidf_matrix = tfidf_matrix
            self.tfidf_vectorizer = tfidf
            
            if os.path.exists(CHROMA_DIR):
                db_files = os.listdir(CHROMA_DIR)
                self.logger.info(f"Созданы файлы БД: {db_files}")
                if not db_files:
                    raise Exception("Файлы БД отсутствуют")
            else:
                raise Exception(f"Папка {CHROMA_DIR} не создана")
            
            self.logger.info(f"Векторное хранилище создано в {CHROMA_DIR}")
            return vector_store
            
        except Exception as e:
            self.logger.error(f"Ошибка создания хранилища: {str(e)}")
            raise

    def _create_rag_chain(self) -> RetrievalQA:
        prompt_template = """Ты - эксперт по анализу документов и поиску информации. Твоя задача - дать максимально точный и полный ответ на вопрос пользователя, используя ТОЛЬКО информацию из предоставленного контекста.

Контекст содержит извлеченные фрагменты из документов. Каждый фрагмент имеет свой источник и может содержать метаданные (например, номер страницы, название листа Excel и т.д.).

Вопрос: {question}

Контекст:
{context}

Инструкции по ответу:
1. Внимательно проанализируй все предоставленные фрагменты контекста.
2. Если информация в разных фрагментах противоречит друг другу, укажи это и объясни возможные причины.
3. Если в контексте есть числовые данные, даты или конкретные значения - используй их точно.
4. Структурируй ответ логически, используя:
   - Маркированные списки для перечислений
   - Подзаголовки для разделения тем
   - Цитаты из контекста в кавычках, если это важно
5. В конце ответа укажи источники информации в формате:
   Источники: [список использованных документов/фрагментов]
6. Если информация в контексте отсутствует или недостаточна - честно признай это.
7. НЕ добавляй информацию, которой нет в контексте.
8. Если вопрос требует уточнения - предложи, какую дополнительную информацию нужно уточнить.

Ответ:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={
                    "k": 8,  # Количество релевантных фрагментов
                    "fetch_k": 20,  # Количество фрагментов для первичного отбора
                    "score_threshold": 0.5  # Порог релевантности
                }
            ),
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": True
            },
            return_source_documents=True
        )

    def multi_way_search(self, query: str, k: int = SEARCH_SETTINGS['initial_k']) -> List[Tuple[LangchainDocument, float]]:
        """Многопроходный поиск с переранжированием"""
        try:
            # Первый проход: семантический поиск
            semantic_results = self.vector_store.similarity_search_with_score(
                query,
                k=k
            )
            
            # Второй проход: полнотекстовый поиск
            query_vector = self.tfidf_vectorizer.transform([query])
            scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Объединяем результаты
            combined_results = []
            for (doc, sem_score), tfidf_score in zip(semantic_results, scores):
                # Нормализуем и комбинируем скоры
                combined_score = (
                    SEARCH_SETTINGS['hybrid_weight'] * (1 - sem_score) +
                    (1 - SEARCH_SETTINGS['hybrid_weight']) * tfidf_score
                )
                
                # Учитываем метаданные документа
                metadata_score = 0.0
                if doc.metadata.get('position') is not None:
                    # Предпочитаем чанки из начала документа
                    metadata_score = 1 - doc.metadata['position']
                
                # Финальный скор с учетом метаданных
                final_score = 0.8 * combined_score + 0.2 * metadata_score
                
                if final_score >= SEARCH_SETTINGS['min_relevance']:
                    combined_results.append((doc, final_score))
            
            # Сортируем по финальному скору
            combined_results.sort(key=lambda x: x[1], reverse=True)
            
            return combined_results[:SEARCH_SETTINGS['rerank_k']]
            
        except Exception as e:
            self.logger.error(f"Ошибка многопроходного поиска: {str(e)}")
            # Возвращаем результаты только семантического поиска в случае ошибки
            return self.vector_store.similarity_search_with_score(query, k=k)

    def analyze_documents(self, query: str) -> dict:
        """Улучшенный анализ документов с многопроходным поиском"""
        try:
            if not self.agent_orchestrator:
                self.logger.error("Агенты не инициализированы")
                return {"error": "Анализатор документов не инициализирован"}
            
            # Очищаем запрос
            clean_query = query.replace("Требуется информация", "").strip()
            self.logger.info(f"Анализ запроса: {clean_query}")
            
            # Проверяем кэш
            query_hash = hashlib.md5(clean_query.encode()).hexdigest()
            cached_response = self._get_cached_response(query_hash)
            if cached_response:
                self.logger.info("Используется кэшированный ответ")
                return cached_response
            
            # Выполняем многопроходный поиск
            search_results = self.multi_way_search(clean_query)
            
            # Формируем контекст с учетом структуры
            context_parts = []
            for doc, score in search_results:
                structure_type = doc.metadata.get("structure", "text")
                source_info = f"Источник: {doc.metadata.get('source', 'Неизвестно')}"
                section_info = f"Раздел: {doc.metadata.get('section', 'Основной текст')}"
                relevance_info = f"Релевантность: {score:.2f}"
                
                if structure_type == "table":
                    context_parts.append(
                        f"{source_info}\n{section_info}\n{relevance_info}\n"
                        f"Тип: Таблица\n{doc.page_content}"
                    )
                elif structure_type == "heading":
                    context_parts.append(
                        f"{source_info}\n{section_info}\n{relevance_info}\n"
                        f"Тип: Заголовок и связанный текст\n{doc.page_content}"
                    )
                else:
                    context_parts.append(
                        f"{source_info}\n{section_info}\n{relevance_info}\n"
                        f"Тип: Текст\n{doc.page_content}"
                    )
            
            context = "\n\n".join(context_parts)
            
            # Получаем ответ через агентов
            response = self.agent_orchestrator.process_query(clean_query, context)
            
            if response["status"] == "error":
                return {"error": response["error"]}
            
            # Сохраняем в кэш
            self._save_to_cache(query_hash, response)
            
            # Форматируем ответ
            result = {
                "answer": {
                    "result": response["result"],
                    "research": response.get("research", ""),
                    "validation": response.get("validation", ""),
                    "query": clean_query,
                    "sources": [
                        {
                            "source": doc.metadata.get("source", "Неизвестно"),
                            "section": doc.metadata.get("section", "Основной текст"),
                            "structure": doc.metadata.get("structure", "text"),
                            "relevance": float(score),
                            "preview": doc.page_content[:200] + "..."
                        }
                        for doc, score in search_results
                    ]
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка при анализе документов: {str(e)}")
            return {"error": str(e)}

    def add_document(self, file_path: str) -> bool:
        try:
            target_path = os.path.join(self.documents_dir, os.path.basename(file_path))
            shutil.copy(file_path, target_path)
            self.read_documents()
            return True
        except Exception as e:
            self.logger.error(f"Ошибка добавления: {str(e)}")
            return False

    def clear_cache(self):
        """Очистка кэша"""
        try:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir)
                self.logger.info("Кэш очищен")
        except Exception as e:
            self.logger.error(f"Ошибка очистки кэша: {str(e)}")

    def _init_rag_chains(self):
        """Инициализация цепочек RAG для индексации и поиска"""
        try:
            # Инициализируем базовые компоненты
            if not self.vector_store:
                self.vector_store = Chroma(
                    collection_name=COLLECTION_NAME,
                    embedding_function=self.embeddings,
                    persist_directory=CHROMA_DIR
                )
            
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

    def _create_retriever(self):
        """Создание улучшенного поисковика с гибридным поиском"""
        try:
            # Базовая конфигурация векторного поисковика
            vector_retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": SEARCH_SETTINGS['initial_k'],
                    "score_threshold": SEARCH_SETTINGS['min_relevance']
                }
            )
            
            # Создаем BM25 поисковик для ключевых слов
            texts = [doc.page_content for doc in self.vector_store.get()]
            self.bm25_retriever = BM25Retriever.from_documents(
                [Document(page_content=text) for text in texts]
            )
            
            # Создаем гибридный поисковик
            hybrid_retriever = HybridSearchRetriever(
                vector_retriever=vector_retriever,
                bm25_retriever=self.bm25_retriever,
                weights={
                    "vector": SEARCH_SETTINGS['hybrid_weight'],
                    "keyword": 1 - SEARCH_SETTINGS['hybrid_weight']
                }
            )
            
            # Добавляем реранжировку результатов
            reranker = ContextualCompressionRetriever(
                base_retriever=hybrid_retriever,
                document_compressor=DocumentCompressorPipeline.from_transformers(
                    embeddings=self.embeddings,
                    base_compressor=EmbeddingsFilter(
                        embeddings=self.embeddings,
                        similarity_threshold=SEARCH_SETTINGS['min_relevance']
                    )
                )
            )
            
            return reranker
            
        except Exception as e:
            self.logger.error(f"Ошибка создания поисковика: {str(e)}")
            return vector_retriever

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

    def _build_context(self, query: str, retrieved_docs: List[Document]) -> str:
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
            retrieved_docs = self.retrieval_chain["retriever"].get_relevant_documents(query)
            
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

    def _load_document(self, file_path: str) -> Dict:
        """Загрузка и первичная обработка документа"""
        try:
            file_type = os.path.splitext(file_path)[1].lower().lstrip('.')
            
            # Извлекаем структуру документа
            structure = extract_document_structure(file_path, file_type)
            
            # Извлекаем текст в зависимости от типа файла
            if file_type == 'pdf':
                # Для PDF используем OCR если нужно
                text = ""
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if not page_text.strip():  # Если текст пустой, пробуем OCR
                            page_text = extract_text_with_ocr(file_path)
                        text += page_text + "\n"
            elif file_type in ['docx', 'doc']:
                doc = Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])
            elif file_type in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
                text = df.to_string()
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            return {
                "text": clean_text(text),
                "structure": structure,
                "metadata": {
                    "source": os.path.basename(file_path),
                    "type": file_type,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки документа {file_path}: {str(e)}")
            raise