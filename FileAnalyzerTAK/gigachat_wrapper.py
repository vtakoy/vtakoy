from typing import Optional, List, Dict, Any, ClassVar
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import Generation, LLMResult
from auth import GigaChatAuth
from pydantic import BaseModel, Field, ConfigDict

class GigaChatWrapper(LLM):
    """Обертка для GigaChat, соответствующая интерфейсу LangChain"""
    
    # Конфигурация для Pydantic v2
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow"
    )
    
    # Атрибуты, которые не будут сериализованы
    _client: GigaChatAuth = None
    
    def __init__(self, client: GigaChatAuth, **kwargs):
        """Инициализация обертки для GigaChat"""
        # Сначала вызываем super().__init__ для инициализации Pydantic полей
        super().__init__(**kwargs)
        # Затем устанавливаем приватные атрибуты
        self._client = client
        
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Вызывает GigaChat API"""
        try:
            response = self._client.chat_completion(prompt)
            return response
        except Exception as e:
            return f"Ошибка при вызове GigaChat: {str(e)}"
            
    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        """Генерирует ответы для списка промптов"""
        generations = []
        for prompt in prompts:
            try:
                response = self._client.chat_completion(prompt)
                generations.append([Generation(text=response)])
            except Exception as e:
                generations.append([Generation(text=f"Ошибка при вызове GigaChat: {str(e)}")])
        return LLMResult(generations=generations)
        
    @property
    def _llm_type(self) -> str:
        """Возвращает тип LLM-модели"""
        return "gigachat" 