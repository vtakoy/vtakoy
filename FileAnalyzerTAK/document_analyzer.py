import os
import shutil
import logging
import base64
import ssl
from typing import List, Dict
from datetime import datetime
from docx import Document
import pandas as pd
from chromadb.config import Settings
from chromadb import PersistentClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from langchain_gigachat.embeddings import GigaChatEmbeddings
from langchain_gigachat.chat_models import GigaChat
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential, retry_if_exception_type
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain_community.vectorstores import Chroma
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(current_dir, "chroma")
COLLECTION_NAME = "documents_collection"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

class GigaChatEmbedder:
    def __init__(self, client):
        credentials = base64.b64encode(
            f"{client.client_id}:{client.client_secret}".encode()
        ).decode()
        
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        self.embeddings = GigaChatEmbeddings(
            credentials=credentials,
            auth_url=client.auth_url,
            base_url=client.api_url,
            verify_ssl_certs=False,
            scope="GIGACHAT_API_PERS",
            timeout=30,
            ssl_context=ssl_context
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)

class DocumentAnalyzer:
    def __init__(self, client, documents_dir=None):
        self.logger = logging.getLogger('DocumentAnalyzer')
        self.logger.setLevel(logging.INFO)
        
        # Проверяем, есть ли уже хэндлеры у логгера
        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            os.makedirs("logs", exist_ok=True)
            
            file_handler = logging.FileHandler(
                os.path.join(current_dir, f"logs/document_analyzer_{datetime.now().strftime('%Y%m%d')}.log"),
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
        
        self.client = client
        self.documents_dir = documents_dir or os.path.join(current_dir, "documents")
        os.makedirs(self.documents_dir, exist_ok=True)
        
        self.logger.info(f"Используется папка документов: {self.documents_dir}")
        self.logger.info(f"Используется папка Chroma: {CHROMA_DIR}")
        
        self._clean_chroma_dir()
        self._init_components()
        self.vector_store = None
        self.rag_chain = None
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
                self.logger.info(f"Успешно проиндексировано {len(documents)} секций")
            else:
                self.logger.warning("Нет документов для индексации")
                
        except Exception as e:
            self.logger.error(f"Ошибка индексации: {str(e)}")
            raise

    def _load_all_documents(self) -> List[LangchainDocument]:
        all_sections = []
        
        for filename in os.listdir(self.documents_dir):
            file_path = os.path.join(self.documents_dir, filename)
            
            try:
                if filename.endswith('.txt'):
                    sections = self._read_txt(file_path)
                elif filename.endswith('.docx'):
                    sections = self._read_word(file_path)
                elif filename.endswith(('.xlsx', '.xls')):
                    sections = self._read_excel(file_path)
                elif filename.endswith('.pdf'):
                    sections = self._read_pdf(file_path)
                else:
                    self.logger.warning(f"Пропущен файл: {filename}")
                    continue
                    
                all_sections.extend(sections)
                self.logger.info(f"Обработан: {filename} ({len(sections)} секций)")
                
            except Exception as e:
                self.logger.error(f"Ошибка файла {filename}: {str(e)}")
                continue
        
        documents = [
            LangchainDocument(
                page_content=section["text"],
                metadata=section["metadata"]
            ) for section in all_sections
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        
        return text_splitter.split_documents(documents)

    def _read_txt(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return [{
            "text": text,
            "metadata": {"source": os.path.basename(file_path)}
        }]

    def _read_word(self, file_path: str) -> List[Dict]:
        try:
            doc = Document(file_path)
            sections = []
            filename = os.path.basename(file_path)
            
            # Извлечение текста из свойств документа
            core_properties = doc.core_properties
            if core_properties.title or core_properties.subject or core_properties.author:
                props_text = []
                if core_properties.title:
                    props_text.append(f"Заголовок: {core_properties.title}")
                if core_properties.subject:
                    props_text.append(f"Тема: {core_properties.subject}")
                if core_properties.author:
                    props_text.append(f"Автор: {core_properties.author}")
                
                sections.append({
                    "text": "\n".join(props_text),
                    "metadata": {"source": filename, "section": "properties"}
                })
            
            # Извлечение основного текста
            paragraphs_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs_text.append(para.text)
            
            # Разбиваем параграфы на логические секции по 10-15 параграфов
            chunk_size = 15
            for i in range(0, len(paragraphs_text), chunk_size):
                chunk = paragraphs_text[i:i+chunk_size]
                sections.append({
                    "text": "\n".join(chunk),
                    "metadata": {"source": filename, "section": f"text_{i//chunk_size+1}"}
                })
            
            # Извлечение таблиц с форматированием
            for i, table in enumerate(doc.tables):
                table_text = []
                for row in table.rows:
                    row_texts = [cell.text.strip() for cell in row.cells]
                    table_text.append(" | ".join(row_texts))
                
                sections.append({
                    "text": f"Таблица {i+1}:\n" + "\n".join(table_text),
                    "metadata": {"source": filename, "section": f"table_{i+1}"}
                })
            
            # Извлечение текста из колонтитулов
            headers_footers = []
            for section in doc.sections:
                if section.header.is_linked_to_previous == False:
                    for paragraph in section.header.paragraphs:
                        if paragraph.text.strip():
                            headers_footers.append(f"Верхний колонтитул: {paragraph.text}")
                
                if section.footer.is_linked_to_previous == False:
                    for paragraph in section.footer.paragraphs:
                        if paragraph.text.strip():
                            headers_footers.append(f"Нижний колонтитул: {paragraph.text}")
            
            if headers_footers:
                sections.append({
                    "text": "\n".join(headers_footers),
                    "metadata": {"source": filename, "section": "headers_footers"}
                })
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Ошибка чтения Word: {str(e)}")
            return []

    def _read_excel(self, file_path: str) -> List[Dict]:
        try:
            filename = os.path.basename(file_path)
            sections = []
            
            # Чтение всех листов Excel файла
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            for sheet_name in sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Считываем весь лист целиком, а не только первые 10 строк
                sections.append({
                    "text": f"Лист '{sheet_name}':\n{df.to_string(index=False)}",
                    "metadata": {"source": filename, "sheet": sheet_name}
                })
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Ошибка чтения Excel: {str(e)}")
            return []

    def _read_pdf(self, file_path: str) -> List[Dict]:
        try:
            import PyPDF2
            
            filename = os.path.basename(file_path)
            sections = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                # Извлекаем метаданные
                metadata = pdf_reader.metadata
                if metadata:
                    meta_text = []
                    if hasattr(metadata, 'title') and metadata.title:
                        meta_text.append(f"Заголовок: {metadata.title}")
                    if hasattr(metadata, 'author') and metadata.author:
                        meta_text.append(f"Автор: {metadata.author}")
                    if hasattr(metadata, 'subject') and metadata.subject:
                        meta_text.append(f"Тема: {metadata.subject}")
                    
                    if meta_text:
                        sections.append({
                            "text": "\n".join(meta_text),
                            "metadata": {"source": filename, "section": "metadata"}
                        })
                
                # Извлекаем текст из каждой страницы
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text.strip():
                        sections.append({
                            "text": text,
                            "metadata": {"source": filename, "page": page_num + 1}
                        })
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Ошибка чтения PDF: {str(e)}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _create_vector_store(self, documents: List[LangchainDocument]) -> Chroma:
        try:
            self._clean_chroma_dir()
            
            # Настройка Chroma для работы в полном офлайн-режиме
            chroma_settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
            
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedder.embeddings,
                persist_directory=CHROMA_DIR,
                collection_name=COLLECTION_NAME,
                client_settings=chroma_settings
            )
            
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    def analyze_documents(self, query: str) -> dict:
        try:
            if not self.rag_chain:
                self.logger.error("RAG chain не инициализирован")
                return {"error": "Анализатор документов не инициализирован. Пожалуйста, сначала загрузите документы."}
            
            # Добавляем задержку перед запросом
            time.sleep(2)  # 2 секунды между запросами
            
            # Очищаем запрос от префикса и лишних пробелов
            clean_query = query.replace("Требуется информация", "").strip()
            self.logger.info(f"Анализ запроса: {clean_query}")
            
            # Получаем ответ от RAG chain
            response = self.rag_chain.invoke({"query": clean_query})
            
            if not response:
                self.logger.warning("Пустой ответ от RAG chain")
                return {"error": "Не удалось получить ответ"}
            
            # Извлекаем результат и источники
            result = response.get("result", "")
            sources = []
            source_details = []
            
            if "source_documents" in response:
                for doc in response["source_documents"]:
                    if hasattr(doc, "metadata"):
                        source_info = {
                            "source": doc.metadata.get("source", "Неизвестный источник"),
                            "section": doc.metadata.get("section", ""),
                            "page": doc.metadata.get("page", ""),
                            "sheet": doc.metadata.get("sheet", "")
                        }
                        source_details.append(source_info)
                        if source_info["source"] not in sources:
                            sources.append(source_info["source"])
            
            # Удаляем дубликаты и формируем список источников
            unique_sources = list(set(sources))
            
            self.logger.info("Успешно получен ответ от RAG chain")
            return {
                "answer": {
                    "result": result,
                    "sources": unique_sources,
                    "source_details": source_details,
                    "query": clean_query
                }
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка при анализе документов: {str(e)}")
            if "429" in str(e):
                return {"error": "Превышен лимит запросов. Пожалуйста, подождите немного и попробуйте снова."}
            return {"error": f"Ошибка при анализе документов: {str(e)}"}

    def add_document(self, file_path: str) -> bool:
        try:
            target_path = os.path.join(self.documents_dir, os.path.basename(file_path))
            shutil.copy(file_path, target_path)
            self.read_documents()
            return True
        except Exception as e:
            self.logger.error(f"Ошибка добавления: {str(e)}")
            return False