"""
Servicios del chatbot.
Maneja la lógica de negocio, gestión de sesiones y orquestación de workflows.
"""
from typing import Dict, Any, List, AsyncIterator

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from server.shared.logger import get_logger
from server.shared.env import env
from server.shared.redis import create_session_store, SessionStore
from server.shared.files import read_text_file

from server.chatbot.workflows import chatbot_workflow

logger = get_logger(__name__)

# ============================================================================
# Instancias globales
# ============================================================================

session_store: SessionStore = create_session_store()
global_system_prompt = read_text_file(env['GLOBAL_SYSTEM_PROMPT_PATH'])


# ============================================================================
# Funciones de serialización
# ============================================================================

def _serialize_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """
    Serializa la lista de mensajes a una lista de diccionarios para almacenar en Redis.

    Args:
        messages: Lista de mensajes de LangChain

    Returns:
        Lista de diccionarios con los mensajes serializados
    """
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


def _deserialize_messages(data: List[Dict[str, Any]]) -> List[Any]:
    """
    Deserializa una lista de diccionarios a lista de mensajes de LangChain.

    Args:
        data: Lista de diccionarios con mensajes serializados

    Returns:
        Lista de mensajes de LangChain
    """
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


# ============================================================================
# Funciones de gestión de sesiones
# ============================================================================

def _get_session_messages(session_id: str) -> List[Any]:
    """
    Obtiene o crea el historial de mensajes de una sesión desde Redis.

    Args:
        session_id: Identificador de la sesión

    Returns:
        Lista de mensajes de LangChain
    """
    try:
        data = session_store.load_session(session_id)

        if data is None:
            messages = [SystemMessage(content=global_system_prompt)]
            _save_session_messages(session_id, messages)
            return messages

        messages = _deserialize_messages(data)
        return messages

    except Exception as e:
        logger.error(f'Error loading session {session_id}: {e}')
        return [SystemMessage(content=global_system_prompt)]


def _save_session_messages(session_id: str, messages: List[Any]):
    """
    Guarda el historial de mensajes en Redis.

    Args:
        session_id: Identificador de la sesión
        messages: Lista de mensajes de LangChain
    """
    try:
        serialized = _serialize_messages(messages)
        session_store.save_session(session_id, serialized)
    except Exception as e:
        logger.error(f'Error saving session {session_id}: {e}')


# ============================================================================
# Funciones públicas del servicio
# ============================================================================

async def process_message_stream(
    user_input: str,
    session_id: str = 'default'
) -> AsyncIterator[str]:
    """
    Procesa un mensaje del usuario y retorna la respuesta en streaming.

    Args:
        user_input: Mensaje del usuario
        session_id: Identificador de la sesión

    Yields:
        str: Fragmentos de la respuesta del chatbot
    """
    try:
        # Cargar historial de mensajes
        messages = _get_session_messages(session_id)

        # Ejecutar workflow del chatbot
        async for chunk in chatbot_workflow(user_input, messages):
            yield chunk

        # Guardar historial actualizado
        _save_session_messages(session_id, messages)

    except Exception as e:
        logger.error(f'Error processing message in session {session_id}: {str(e)}')
        yield f'Sorry, an error occurred while processing your message: {str(e)}'


def get_history(session_id: str = 'default') -> List[Dict[str, Any]]:
    """
    Obtiene el historial de mensajes de una sesión.
    Solo retorna mensajes de tipo 'human' y 'ai'.

    Args:
        session_id: Identificador de la sesión

    Returns:
        Lista de mensajes serializados (solo human y ai)
    """
    try:
        data = session_store.load_session(session_id)

        if data is None:
            return []

        # Filtrar solo mensajes de tipo 'human' y 'ai'
        filtered_messages = [
            msg for msg in data
            if msg.get('type') in ['human', 'ai']
        ]

        return filtered_messages

    except Exception as e:
        logger.error(f'Error getting history for session {session_id}: {e}')
        return []


def clear_history(session_id: str = 'default'):
    """
    Limpia el historial de una sesión, reiniciándola con el prompt del sistema.

    Args:
        session_id: Identificador de la sesión
    """
    try:
        messages = [SystemMessage(content=global_system_prompt)]
        _save_session_messages(session_id, messages)
        logger.info(f'History cleared for session {session_id}')
    except Exception as e:
        logger.error(f'Error clearing history for session {session_id}: {e}')
