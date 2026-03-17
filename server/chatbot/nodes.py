"""
Nodos especializados para el chatbot.
Cada nodo hereda de las clases base en shared/nodes.py
"""
from typing import AsyncIterator, List, Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase

from server.shared.logger import get_logger
from server.shared.nodes import BaseRunnableNode, BaseRunnableStreamNode, BaseRunnableAgentNode

logger = get_logger(__name__)


class RouterNode(BaseRunnableAgentNode):
    """
    Clasifica la intención del usuario y determina la ruta a seguir.
    Opciones: 'offside', 'chat', 'sql'
    """

    def __init__(self, llm: ChatOllama, system_prompt: str):
        super().__init__(llm)
        self.system_prompt = system_prompt

    async def run(self, user_input: str) -> str:
        """
        Clasifica entrada del usuario y retorna ruta correspondiente.

        Args:
            user_input: Pregunta del usuario

        Returns:
            str: Una de las siguientes opciones: 'offside', 'chat', 'sql'
        """
        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_input)
            ]

            response = await self.llm.ainvoke(messages)
            route = response.content.strip().lower().strip('"').strip("'")

            if route not in ['offside', 'chat', 'sql']:
                logger.warning(f'Router returned unexpected value: "{route}", defaulting to "chat"')
                return 'chat'

            logger.info(f'Router determined route: "{route}"')
            return route

        except Exception as e:
            logger.error(f'Error in RouterNode: {e}')
            return 'chat'


class OffsideNode(BaseRunnableStreamNode):
    """
    Maneja preguntas fuera del dominio con una respuesta predefinida.
    """

    async def run_stream(self) -> AsyncIterator[str]:
        """
        Retorna respuesta predefinida para preguntas fuera del dominio.
        """
        response = (
            "I'm sorry, that question is outside my area of knowledge. "
            "I can help you with general conversations. "
            "Is there anything else I can assist you with?"
        )
        yield response


class ChatNode(BaseRunnableStreamNode):
    """
    Nodo para conversación general con el modelo.
    """

    def __init__(self, llm: ChatOllama):
        self.llm = llm

    async def run_stream(self, user_input: str, messages: List[Any]) -> AsyncIterator[str]:
        """
        Procesa conversación general en streaming.

        Args:
            user_input: Pregunta del usuario
            messages: Historial de mensajes de la conversación

        Yields:
            str: Fragmentos de la respuesta del modelo
        """
        try:
            messages.append(HumanMessage(content=user_input))

            full_response = ''

            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    yield chunk.content

            messages.append(AIMessage(content=full_response))

        except Exception as e:
            logger.error(f'Error in ChatNode streaming: {e}')
            yield f'Sorry, an error occurred while processing your message: {str(e)}'


class QueryBuilderNode(BaseRunnableAgentNode):
    """
    Construye consultas SQL desde lenguaje natural.
    """

    def __init__(self, llm: ChatOllama, db_url: str, system_prompt: str, prompt_template: str):
        super().__init__(llm)
        self.db = SQLDatabase.from_uri(db_url)
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template

    async def run(self, user_input: str) -> str:
        """
        Construye query SQL desde pregunta en lenguaje natural.

        Args:
            user_input: Pregunta del usuario

        Returns:
            str: Query SQL generado

        Raises:
            Exception: Si el modelo no genera una consulta válida
        """
        try:
            table_info = self.db.get_table_info()

            prompt = self.prompt_template.format(
                system_prompt=self.system_prompt,
                table_info=table_info,
                user_input=user_input
            )

            messages = [SystemMessage(content=prompt)]

            response = await self.llm.ainvoke(messages)

            logger.info(f'Respuesta raw del modelo: {response.content}')

            query = response.content.strip()

            # Limpiar markdown SQL si existe
            if query.startswith('```sql'):
                query = query.replace('```sql', '').replace('```', '').strip()
            elif query.startswith('```'):
                query = query.replace('```', '').strip()

            if not query:
                error_msg = f'El modelo QueryBuilder devolvió una respuesta vacía. Respuesta original: "{response.content}"'
                logger.error(error_msg)
                raise Exception('Error crítico: El modelo no generó una consulta SQL válida. Verifica la configuración del modelo y el prompt.')

            logger.info(f'Query SQL construido: {query}')
            return query

        except Exception as e:
            logger.error(f'Error in QueryBuilderNode: {e}')
            raise Exception(f'Error al construir la consulta SQL: {str(e)}')


class QueryExecutorNode(BaseRunnableNode):
    """
    Ejecuta consultas SQL y retorna resultados.
    """

    def __init__(self, db_url: str):
        self.db = SQLDatabase.from_uri(db_url)

    async def run(self, query: str) -> str:
        """
        Ejecuta query SQL y retorna resultados.

        Args:
            query: Query SQL a ejecutar

        Returns:
            str: Resultados de la consulta

        Raises:
            Exception: Si hay error en la ejecución
        """
        try:
            if not query or query.strip() == '':
                error_msg = f'Se intentó ejecutar un query vacío o nulo: "{query}"'
                logger.error(error_msg)
                raise Exception('Error crítico: No se puede ejecutar un query SQL vacío')

            logger.info(f'Ejecutando query SQL: {query}')
            result = self.db.run(query)

            logger.info(f'Query ejecutado exitosamente. Resultados: {result}')
            return result

        except Exception as e:
            logger.error(f'Error in QueryExecutorNode: {e}')
            raise Exception(f'Error al ejecutar la consulta: {str(e)}')


class SQLInterpreterNode(BaseRunnableStreamNode):
    """
    Interpreta resultados SQL y genera respuestas en lenguaje natural.
    """

    def __init__(self, llm: ChatOllama, prompt_template: str, system_message: SystemMessage):
        self.llm = llm
        self.prompt_template = prompt_template
        self.system_message = system_message

    async def run_stream(
        self,
        user_input: str,
        sql_query: str,
        query_results: str
    ) -> AsyncIterator[str]:
        """
        Interpreta resultados SQL y genera respuesta en lenguaje natural.

        Args:
            user_input: Pregunta original del usuario
            sql_query: Query SQL ejecutado
            query_results: Resultados de la consulta

        Yields:
            str: Fragmentos de la respuesta interpretada
        """
        try:
            if not query_results or query_results.strip() == '':
                query_results = 'No data available.'

            interpreter_context = self.prompt_template.format(
                question=user_input,
                data=query_results,
                sql_query=sql_query
            )

            temp_messages = [
                self.system_message,
                SystemMessage(content=interpreter_context),
                HumanMessage(content=user_input)
            ]

            async for chunk in self.llm.astream(temp_messages):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content

        except Exception as e:
            logger.error(f'Error in SQLInterpreterNode: {e}')
            yield f'Sorry, an error occurred while interpreting the results: {str(e)}'
