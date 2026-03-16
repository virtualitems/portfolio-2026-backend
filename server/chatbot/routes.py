from fastapi import APIRouter, Form, Request, HTTPException
from fastapi.responses import Response, StreamingResponse

from ..shared.logger import get_logger
from ..shared.redis import create_session_store
from .agent import chatbot_agent

logger = get_logger(__name__)
router = APIRouter(prefix='/api/chatbot', tags=['chatbot'])

# Crear instancia del session store
session_store = create_session_store()


@router.post('/text-to-text')
async def chat_stream(request: Request, q: str = Form(..., description='Pregunta para el modelo')):
    """
    Endpoint para chatear con el agente en modo streaming.
    """
    try:
        session_id = request.headers.get('X-Session-Id', 'default')

        return StreamingResponse(
            chatbot_agent.invoke_stream(q, session_id=session_id),
            media_type='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
            }
        )
    except Exception as e:
        logger.error(f'Error in chat: {str(e)}')
        return Response(
            content=f'Error: {str(e)}',
            status_code=500,
            media_type='text/plain'
        )


@router.get('/history')
async def get_chat_history(request: Request):
    """
    Endpoint para obtener el historial de mensajes de una sesión directamente desde Redis.
    Útil para restaurar el historial después de recargar la página.
    Solo devuelve mensajes de tipo 'human' y 'ai'.
    """
    try:
        session_id = request.headers.get('X-Session-Id', 'default')

        # Obtener datos directamente desde Redis
        messages_data = session_store.load_session(session_id)

        if messages_data is None:
            return {
                'data': []
            }

        # Filtrar solo mensajes de tipo 'human' y 'ai'
        filtered_messages = [
            msg for msg in messages_data
            if msg.get('type') in ['human', 'ai']
        ]

        return {
            'data': filtered_messages
        }
    except Exception as e:
        logger.error(f'Error getting history: {str(e)}')
        return Response(
            content=f'Error: {str(e)}',
            status_code=500,
            media_type='application/json'
        )


@router.post('/clear-history')
async def clear_chat_history(request: Request):
    """
    Endpoint para limpiar el historial de chat de una sesión.
    """
    try:
        session_id = request.headers.get('X-Session-Id', 'default')
        chatbot_agent.clear_history(session_id)
        return {'status': 'success', 'message': f'History cleared for session {session_id}'}
    except Exception as e:
        logger.error(f'Error clearing history: {str(e)}')
        return Response(
            content=f'Error: {str(e)}',
            status_code=500,
            media_type='application/json'
        )
