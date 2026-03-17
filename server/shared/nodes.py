from __future__ import annotations

from typing import AsyncIterator, Union
from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama

class BaseAgentNode(ABC):
    """
    Nodo base para los agentes.
    """

    def __init__(self, llm: ChatOllama, *args, **kwargs):
        self.llm = llm


class BaseRunnableAgentNode(BaseAgentNode):
    """
    Nodo que puede procesar un input y generar una respuesta.
    """
    @abstractmethod
    async def run(self, *args, **kwargs) -> str:
        """
        Procesa el input y retorna la respuesta.
        """
        pass

class BaseRunnableStreamAgentNode(BaseAgentNode):
    """
    Nodo que puede procesar un input y generar una respuesta en streaming.
    """
    @abstractmethod
    async def run_stream(self, *args, **kwargs) -> AsyncIterator[str]:
        """
        Procesa el input y retorna la respuesta como un stream de texto.
        """
        pass


class BaseRunnableNode:
    """
    Nodo que puede procesar un input y generar una respuesta.
    """
    @abstractmethod
    async def run(self, *args, **kwargs) -> str:
        """
        Procesa el input y retorna la respuesta.
        """
        pass

class BaseRunnableStreamNode:
    """
    Nodo que puede procesar un input y generar una respuesta en streaming.
    """
    @abstractmethod
    async def run_stream(self, *args, **kwargs) -> AsyncIterator[str]:
        """
        Procesa el input y retorna la respuesta como un stream de texto.
        """
        pass
