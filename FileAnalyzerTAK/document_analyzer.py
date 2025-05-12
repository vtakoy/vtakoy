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
from tenacity import retry, stop_after_attempt, wait_fixed
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain_community.vectorstores import Chroma

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
            
            for para in doc.paragraphs:
                if para.text.strip():
                    sections.append({
                        "text": para.text,
                        "metadata": {"source": filename}
                    })
                    
            for table in doc.tables:
                table_text = "\n".join(" | ".join(cell.text for cell in row.cells) for row in table.rows)
                sections.append({
                    "text": f"Таблица:\n{table_text}",
                    "metadata": {"source": filename}
                })
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Ошибка чтения Word: {str(e)}")
            return []

    def _read_excel(self, file_path: str) -> List[Dict]:
        try:
            df = pd.read_excel(file_path)
            filename = os.path.basename(file_path)
            sections = []
            
            sections.append({
                "text": f"Данные:\n{df.head(10).to_string()}",
                "metadata": {"source": filename}
            })
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Ошибка чтения Excel: {str(e)}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _create_vector_store(self, documents: List[LangchainDocument]) -> Chroma:
        try:
            self._clean_chroma_dir()
            
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedder.embeddings,
                persist_directory=CHROMA_DIR,
                collection_name=COLLECTION_NAME
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
        prompt_template = """Ты - эксперт по документам. Ответь на вопрос используя контекст:
        Вопрос: {question}
        Контекст: {context}
        Ответ:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def analyze_documents(self, query: str) -> str:
        try:
            clean_query = query.replace("Требуется информация", "").strip()
            self.logger.info(f"Обработка: {clean_query}")
            
            if not self.vector_store:
                raise ValueError("Хранилище не инициализировано")
                
            response = self.rag_chain.invoke({"query": clean_query})
            return response.get("result", "Ответ не найден")
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа: {str(e)}")
            return f"Ошибка: {str(e)}"

    def add_document(self, file_path: str) -> bool:
        try:
            target_path = os.path.join(self.documents_dir, os.path.basename(file_path))
            shutil.copy(file_path, target_path)
            self.read_documents()
            return True
        except Exception as e:
            self.logger.error(f"Ошибка добавления: {str(e)}")
            return False