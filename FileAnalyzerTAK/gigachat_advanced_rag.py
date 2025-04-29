from gigachat_langchain import GigaChatLangchain
from langchain_core.documents import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
import os
import time
import pandas as pd
from docx import Document
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

def main():
    # Инициализируем GigaChatLangchain
    print("Инициализация GigaChatLangchain...")
    gigachat = GigaChatLangchain(verify_ssl=False)
    
    # Получаем модель эмбеддингов и чатовую модель
    embeddings = gigachat.get_embeddings()
    llm = gigachat.get_chat_model()
    
    # Создаем загрузку документов
    documents_dir = "documents"
    documents = load_documents(documents_dir)
    
    # Создаем векторное хранилище
    print("Создание векторного хранилища...")
    db = create_vector_store(documents, embeddings)
    
    # Создаем RAG-цепочку с инструктированием поиска
    print("Создание RAG-цепочки с улучшенным поиском...")
    qa_chain = create_rag_chain(db, llm)
    
    # Демонстрация запросов и получения ответов
    demo_questions = [
        "Требуется информация о способах доставки заказа",
        "Требуется информация о минимальной сумме заказа",
        "Требуется информация о контактах для получения помощи"
    ]
    
    print("\n=== Расширенная демонстрация RAG на основе GigaChat ===\n")
    
    for question in demo_questions:
        print(f"\nВопрос: {question}")
        
        # Задержка, чтобы не превысить лимиты API
        time.sleep(1)
        
        # Получаем ответ от RAG
        start_time = time.time()
        response = qa_chain.invoke({"query": question})
        end_time = time.time()
        
        print(f"\nОтвет: {response['result']}")
        print(f"Время обработки: {end_time - start_time:.2f} секунд")
        print("-" * 50)

def read_txt(file_path: str) -> List[Dict[str, Any]]:
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
    
    print(f"Из файла {os.path.basename(file_path)} извлечено {len(sections)} секций")
    return sections

def read_word(file_path: str) -> List[Dict[str, Any]]:
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
            
            # Получаем данные таблицы в виде строк для лучшей читаемости
            rows_data = []
            for i, row in enumerate(table.rows):
                cells_data = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells_data:
                    row_text = " | ".join(cells_data)
                    rows_data.append(row_text)
            
            # Форматируем данные таблицы для лучшего извлечения
            if len(rows_data) > 0:
                # Если первая строка короче (возможно, заголовок), обрабатываем особо
                if len(rows_data) > 1 and len(rows_data[0].split('|')) < len(rows_data[1].split('|')):
                    table_text += f"Заголовок таблицы: {rows_data[0]}\n\n"
                    for row_idx, row in enumerate(rows_data[1:], 1):
                        table_text += f"Строка {row_idx}: {row}\n"
                else:
                    for row_idx, row in enumerate(rows_data, 1):
                        table_text += f"Строка {row_idx}: {row}\n"
            
            # Добавляем содержимое таблицы как отдельную секцию
            if table_text.strip() != f"Таблица {table_idx + 1}:\n\n":
                table_section = {
                    "text": table_text,
                    "metadata": {
                        "source": filename,
                        "section": f"Таблица {table_idx + 1}"
                    }
                }
                sections.append(table_section)
            
        print(f"Из файла {filename} извлечено {len(sections)} секций")
        return sections
    except Exception as e:
        print(f"Ошибка при чтении Word файла: {str(e)}")
        return []

def read_excel(file_path: str) -> List[Dict[str, Any]]:
    """Читает Excel файл и возвращает список словарей с данными"""
    try:
        df = pd.read_excel(file_path)
        sections = []
        
        # Добавляем имя файла в виде метаданных
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
        
        print(f"Из файла {filename} извлечено {len(sections)} секций")
        return sections
    except Exception as e:
        print(f"Ошибка при чтении Excel файла: {str(e)}")
        return []

def load_documents(documents_dir):
    """Загружает документы из указанной директории"""
    all_sections = []
    
    for filename in os.listdir(documents_dir):
        file_path = os.path.join(documents_dir, filename)
        
        if filename.endswith('.txt'):
            sections = read_txt(file_path)
        elif filename.endswith('.docx'):
            sections = read_word(file_path)
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            sections = read_excel(file_path)
        else:
            print(f"Пропуск файла {filename}: неподдерживаемый формат")
            continue
        
        all_sections.extend(sections)
    
    # Создаем LangChain документы
    documents = []
    for section in all_sections:
        doc = LangchainDocument(
            page_content=section["text"],
            metadata=section["metadata"]
        )
        documents.append(doc)
    
    # Разделяем документы на чанки для лучшего поиска
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    
    # Разбиваем документы на чанки
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Загружено {len(documents)} секций, создано {len(chunked_documents)} чанков")
    
    return chunked_documents

def create_vector_store(documents, embeddings):
    """Создает векторное хранилище из документов"""
    # Проверяем наличие директории для Chroma и удаляем ее, если она существует
    if os.path.exists("chroma_advanced_db"):
        import shutil
        shutil.rmtree("chroma_advanced_db")
        print("Удаление старой базы Chroma")
    
    # Создаем векторное хранилище Chroma
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="chroma_advanced_db"
    )
    
    # Сохраняем векторное хранилище
    vector_store.persist()
    print(f"Векторное хранилище создано с {len(documents)} документами")
    
    return vector_store

def create_rag_chain(vector_store, llm):
    """Создает RAG-цепочку для ответов на вопросы с улучшенным промптом"""
    # Шаблон промпта с инструкциями для извлечения информации
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
    
    # Создаем retriever с настройками для расширенного поиска
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 6,  # Увеличиваем количество документов для большего контекста
            "search_type": "similarity"
        }
    )
    
    # Создаем класс InstructionRetriever для улучшения поиска
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
        llm=llm,
        chain_type="stuff",  # Используем "stuff" для объединения всех чанков в один контекст
        retriever=instructed_retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True  # Возвращаем исходные документы для отладки
    )

if __name__ == "__main__":
    main() 