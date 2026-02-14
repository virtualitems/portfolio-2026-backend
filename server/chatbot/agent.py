"""
Agente de chatbot usando LangChain y Ollama.
"""
import json
from typing import Dict, Any, List, AsyncIterator

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent

from ..shared.logger import get_logger
from ..shared.env import env
from ..shared.redis import create_session_store, SessionStore
from ..shared.files import read_text_file
from ..shared.database import database

logger = get_logger(__name__)


class RouterNode:
    """Clasifica intención del usuario: 'offside', 'chat', o 'sql'"""

    def __init__(self, llm: ChatOllama):
        self.llm = llm
        self.system_prompt = read_text_file(env['ROUTER_NODE_SYSTEM_PROMPT_PATH'])

    async def route(self, user_input: str) -> str:
        """Clasifica entrada del usuario y retorna ruta"""
        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_input)
            ]

            response = await self.llm.ainvoke(messages)
            route = response.content.strip().lower().strip('\"').strip('\'')

            if route not in ['offside', 'chat', 'sql']:
                logger.warning(f'Router returned unexpected value: "{route}", defaulting to "chat"')
                return 'chat'

            logger.info(f'Router determined route: "{route}"')
            return route

        except Exception as e:
            logger.error(f'Error in RouterNode: {e}')
            return 'chat'


class OffsideNode:
    """Maneja preguntas fuera del dominio"""

    @staticmethod
    async def respond_stream() -> AsyncIterator[str]:
        """Retorna respuesta predefinida para preguntas fuera del dominio"""
        response = ('I\'m sorry, that question is outside my area of knowledge. '
                    'I can help you with general conversations. '
                    'Is there anything else I can assist you with?')

        yield response


class ChatNode:
    """Conversación general con el modelo"""

    def __init__(self, llm: ChatOllama):
        self.llm = llm

    async def chat_stream(self, user_input: str, messages: List[Any]) -> AsyncIterator[str]:
        """Procesa conversación general en streaming"""
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
            yield f'Lo siento, hubo un error al procesar tu mensaje: {str(e)}'

class QueryBuilderNode:
    """Construye consultas SQL desde lenguaje natural"""

    def __init__(self, llm: ChatOllama, db_url: str):
        self.llm = llm
        self.db = SQLDatabase.from_uri(db_url)
        self.system_prompt = read_text_file(env['QUERY_BUILDER_NODE_SYSTEM_PROMPT_PATH'])
        self.prompt_template = read_text_file(env['QUERY_BUILDER_PROMPT_TEMPLATE_PATH'])

    async def build_query(self, user_input: str) -> str:
        """Construye query SQL desde pregunta en lenguaje natural"""
        try:
            table_info = self.db.get_table_info()

            prompt = self.prompt_template.format(
                system_prompt=self.system_prompt,
                table_info=table_info,
                user_input=user_input
            )

            messages = [
                SystemMessage(content=prompt)
            ]

            response = await self.llm.ainvoke(messages)

            logger.info(f'Respuesta raw del modelo: {response.content}')

            query = response.content.strip()

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


class QueryExecutorNode:
    """Ejecuta consultas SQL y retorna resultados"""

    def __init__(self, db_url: str):
        self.db = SQLDatabase.from_uri(db_url)

    async def execute_query(self, query: str) -> str:
        """Ejecuta query SQL y retorna resultados"""
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


class ChatbotAgent:
    """Agente coordinador que usa sistema de nodos para procesar peticiones"""

    def __init__(self):
        self.router_model_name = env['ROUTER_LLM_MODEL_NAME']
        self.router_temperature = float(env.get('ROUTER_TEMPERATURE', '0.0'))

        self.chat_model_name = env['CHAT_LLM_MODEL_NAME']
        self.chat_temperature = float(env.get('CHAT_TEMPERATURE', '0.7'))

        self.query_builder_model_name = env['QUERY_BUILDER_LLM_MODEL_NAME']
        self.query_builder_temperature = float(env.get('QUERY_BUILDER_TEMPERATURE', '0.0'))

        self.query_executor_model_name = env['QUERY_EXECUTOR_LLM_MODEL_NAME']
        self.query_executor_temperature = float(env.get('QUERY_EXECUTOR_TEMPERATURE', '0.0'))

        self.base_url = env['OLLAMA_BASE_URL']

        self.session_store: SessionStore = create_session_store()

        self.router_llm = ChatOllama(
            model=self.router_model_name,
            temperature=self.router_temperature,
            base_url=self.base_url,
        )

        self.chat_llm = ChatOllama(
            model=self.chat_model_name,
            temperature=self.chat_temperature,
            base_url=self.base_url,
        )

        self.query_builder_llm = ChatOllama(
            model=self.query_builder_model_name,
            temperature=self.query_builder_temperature,
            base_url=self.base_url,
        )

        self.router_node = RouterNode(self.router_llm)
        self.offside_node = OffsideNode()
        self.chat_node = ChatNode(self.chat_llm)
        self.query_builder_node = QueryBuilderNode(self.query_builder_llm, database.db_url)
        self.query_executor_node = QueryExecutorNode(database.db_url)

        self.system_prompt = read_text_file(env['GLOBAL_SYSTEM_PROMPT_PATH'])

        self.sql_interpreter_prompt_template = read_text_file(env['SQL_INTERPRETER_PROMPT_TEMPLATE_PATH'])

    def _serialize_messages(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Serializa la lista de mensajes a una lista de diccionarios"""
        serialized = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                serialized.append({'type': 'system', 'content': msg.content})
            elif isinstance(msg, HumanMessage):
                serialized.append({'type': 'human', 'content': msg.content})
            elif isinstance(msg, AIMessage):
                msg_data = {'type': 'ai', 'content': msg.content}
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    msg_data['tool_calls'] = msg.tool_calls
                serialized.append(msg_data)
            elif isinstance(msg, ToolMessage):
                serialized.append({
                    'type': 'tool',
                    'content': msg.content,
                    'tool_call_id': msg.tool_call_id
                })
        return serialized

    def _deserialize_messages(self, data: List[Dict[str, Any]]) -> List[Any]:
        """Deserializa una lista de diccionarios a lista de mensajes"""
        messages = []
        for msg_data in data:
            msg_type = msg_data['type']
            if msg_type == 'system':
                messages.append(SystemMessage(content=msg_data['content']))
            elif msg_type == 'human':
                messages.append(HumanMessage(content=msg_data['content']))
            elif msg_type == 'ai':
                ai_msg = AIMessage(content=msg_data['content'])
                if 'tool_calls' in msg_data:
                    ai_msg.tool_calls = msg_data['tool_calls']
                messages.append(ai_msg)
            elif msg_type == 'tool':
                messages.append(ToolMessage(
                    content=msg_data['content'],
                    tool_call_id=msg_data['tool_call_id']
                ))
        return messages

    def _get_session_messages(self, session_id: str) -> List[Any]:
        """Obtiene o crea historial de mensajes desde Redis"""
        try:
            data = self.session_store.load_session(session_id)

            if data is None:
                messages = [SystemMessage(content=self.system_prompt)]
                self._save_session_messages(session_id, messages)
                return messages

            messages = self._deserialize_messages(data)
            return messages

        except Exception as e:
            logger.error(f'Error loading session {session_id}: {e}')
            return [SystemMessage(content=self.system_prompt)]

    def _save_session_messages(self, session_id: str, messages: List[Any]):
        """Guarda historial de mensajes en Redis"""
        try:
            serialized = self._serialize_messages(messages)
            self.session_store.save_session(session_id, serialized)
        except Exception as e:
            logger.error(f'Error saving session {session_id}: {e}')

    async def invoke_stream(self, user_input: str, session_id: str = 'default') -> AsyncIterator[str]:
        """Procesa mensaje del usuario y retorna respuesta en streaming"""
        try:
            messages = self._get_session_messages(session_id)

            route = await self.router_node.route(user_input)
            print(route)

            if route == 'offside':
                messages.append(HumanMessage(content=user_input))
                full_response = ''
                async for chunk in self.offside_node.respond_stream():
                    full_response += chunk
                    yield chunk
                messages.append(AIMessage(content=full_response))

            elif route == 'sql':
                try:
                    messages.append(HumanMessage(content=user_input))

                    sql_query = await self.query_builder_node.build_query(user_input)

                    query_results = await self.query_executor_node.execute_query(sql_query)

                    if not query_results or query_results.strip() == '':
                        query_results = 'No data available.'

                    interpreter_context = self.sql_interpreter_prompt_template.format(
                        question=user_input,
                        data=query_results
                    )

                    temp_messages = [
                        messages[0],
                        HumanMessage(content=interpreter_context)
                    ]

                    full_response = ''
                    async for chunk in self.chat_llm.astream(temp_messages):
                        if hasattr(chunk, 'content') and chunk.content:
                            full_response += chunk.content
                            yield chunk.content

                    messages.append(AIMessage(content=full_response))

                except Exception as e:
                    logger.error(f'Error in SQL flow: {e}')
                    error_message = f'Lo siento, hubo un error al procesar tu consulta a la base de datos: {str(e)}'
                    messages.append(AIMessage(content=error_message))
                    yield error_message

            else:
                async for chunk in self.chat_node.chat_stream(user_input, messages):
                    yield chunk

            self._save_session_messages(session_id, messages)

        except Exception as e:
            logger.error(f'Error in agent invocation: {str(e)}')
            yield f'Lo siento, hubo un error al procesar tu mensaje: {str(e)}'

    def get_history(self, session_id: str = 'default') -> List[Any]:
        """Obtiene historial de una sesión"""
        return self._get_session_messages(session_id)

    def clear_history(self, session_id: str = 'default'):
        """Limpia historial de una sesión"""
        try:
            messages = [SystemMessage(content=self.system_prompt)]
            self._save_session_messages(session_id, messages)
        except Exception as e:
            logger.error(f'Error clearing history for session {session_id}: {e}')


chatbot_agent = ChatbotAgent()
