from typing import List, Dict, Optional
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import Tool
from langchain_gigachat.chat_models import GigaChat
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import logging
from datetime import datetime

logger = logging.getLogger('DocumentAgents')

class BaseAgent:
    def __init__(self, llm: GigaChat, name: str, description: str):
        self.llm = llm
        self.name = name
        self.description = description
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.tools = []
        self._setup_tools()
        self._setup_agent()

    def _setup_tools(self):
        """Настройка инструментов агента"""
        pass

    def _setup_agent(self):
        """Настройка агента с инструментами"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Ты - {self.name}. {self.description}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )

    def run(self, query: str) -> Dict:
        """Выполнение запроса агентом"""
        try:
            result = self.agent_executor.invoke({"input": query})
            return {
                "status": "success",
                "result": result["output"],
                "agent": self.name
            }
        except Exception as e:
            logger.error(f"Ошибка агента {self.name}: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "agent": self.name
            }

class ResearchAgent(BaseAgent):
    """Агент для поиска и анализа информации в документах"""
    def __init__(self, llm: GigaChat, vector_store):
        super().__init__(
            llm=llm,
            name="Research Agent",
            description="Ты - эксперт по поиску и анализу информации в документах. "
                       "Твоя задача - находить релевантную информацию и структурировать её."
        )
        self.vector_store = vector_store
        self._setup_tools()

    def _setup_tools(self):
        self.tools = [
            Tool(
                name="search_documents",
                func=self.vector_store.similarity_search,
                description="Поиск релевантной информации в документах"
            ),
            Tool(
                name="analyze_context",
                func=self._analyze_context,
                description="Анализ найденного контекста"
            )
        ]

    def _analyze_context(self, context: str) -> str:
        """Анализ контекста и выделение ключевой информации"""
        prompt = f"""Проанализируй следующий контекст и выдели ключевую информацию:
        
        {context}
        
        Выдели:
        1. Основные факты
        2. Важные детали
        3. Возможные противоречия
        4. Недостающую информацию
        """
        
        response = self.llm.invoke(prompt)
        return response.content

class WriterAgent(BaseAgent):
    """Агент для формирования структурированных ответов"""
    def __init__(self, llm: GigaChat):
        super().__init__(
            llm=llm,
            name="Writer Agent",
            description="Ты - эксперт по составлению структурированных ответов. "
                       "Твоя задача - формировать четкие и информативные ответы на основе найденной информации."
        )
        self._setup_tools()

    def _setup_tools(self):
        self.tools = [
            Tool(
                name="format_response",
                func=self._format_response,
                description="Форматирование ответа с учетом структуры и стиля"
            ),
            Tool(
                name="check_consistency",
                func=self._check_consistency,
                description="Проверка согласованности информации"
            )
        ]

    def _format_response(self, content: str) -> str:
        """Форматирование ответа"""
        prompt = f"""Отформатируй следующий ответ, сделав его структурированным и понятным:
        
        {content}
        
        Используй:
        - Подзаголовки для разделения тем
        - Маркированные списки для перечислений
        - Цитаты для важных утверждений
        - Таблицы для структурированных данных
        """
        
        response = self.llm.invoke(prompt)
        return response.content

    def _check_consistency(self, content: str) -> str:
        """Проверка согласованности информации"""
        prompt = f"""Проверь следующий текст на согласованность и логичность:
        
        {content}
        
        Проверь:
        1. Нет ли противоречий
        2. Логична ли структура
        3. Полнота информации
        4. Связность изложения
        """
        
        response = self.llm.invoke(prompt)
        return response.content

class ValidatorAgent(BaseAgent):
    """Агент для проверки и валидации ответов"""
    def __init__(self, llm: GigaChat):
        super().__init__(
            llm=llm,
            name="Validator Agent",
            description="Ты - эксперт по проверке и валидации информации. "
                       "Твоя задача - проверять точность и достоверность ответов."
        )
        self._setup_tools()

    def _setup_tools(self):
        self.tools = [
            Tool(
                name="validate_facts",
                func=self._validate_facts,
                description="Проверка фактической точности"
            ),
            Tool(
                name="check_sources",
                func=self._check_sources,
                description="Проверка источников информации"
            )
        ]

    def _validate_facts(self, content: str) -> str:
        """Проверка фактической точности"""
        prompt = f"""Проверь следующие утверждения на фактическую точность:
        
        {content}
        
        Проверь:
        1. Соответствие источникам
        2. Актуальность информации
        3. Корректность интерпретации
        4. Наличие необоснованных выводов
        """
        
        response = self.llm.invoke(prompt)
        return response.content

    def _check_sources(self, content: str) -> str:
        """Проверка источников информации"""
        prompt = f"""Проверь качество и надежность источников информации:
        
        {content}
        
        Проверь:
        1. Достаточность источников
        2. Надежность источников
        3. Релевантность источников
        4. Актуальность источников
        """
        
        response = self.llm.invoke(prompt)
        return response.content

class AgentOrchestrator:
    """Оркестратор для управления агентами"""
    def __init__(self, llm: GigaChat, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        self.research_agent = ResearchAgent(llm, vector_store)
        self.writer_agent = WriterAgent(llm)
        self.validator_agent = ValidatorAgent(llm)
        self.logger = logging.getLogger('AgentOrchestrator')

    def process_query(self, query: str) -> Dict:
        """Обработка запроса через цепочку агентов"""
        try:
            # Шаг 1: Поиск и анализ информации
            research_result = self.research_agent.run(query)
            if research_result["status"] == "error":
                return research_result

            # Шаг 2: Формирование ответа
            writer_result = self.writer_agent.run(research_result["result"])
            if writer_result["status"] == "error":
                return writer_result

            # Шаг 3: Валидация ответа
            validation_result = self.validator_agent.run(writer_result["result"])
            if validation_result["status"] == "error":
                return validation_result

            # Формируем финальный ответ
            return {
                "status": "success",
                "result": validation_result["result"],
                "research": research_result["result"],
                "validation": validation_result["result"]
            }

        except Exception as e:
            self.logger.error(f"Ошибка обработки запроса: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            } 