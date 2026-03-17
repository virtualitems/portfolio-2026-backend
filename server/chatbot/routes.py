from fastapi import APIRouter, Form, Request, HTTPException, status
from fastapi.responses import Response, StreamingResponse

from server.shared.logger import get_logger
from server.chatbot import services

logger = get_logger(__name__)
router = APIRouter(prefix='/chatbot', tags=['chatbot'])


def get_session_id(request: Request) -> str:
    """
    Valida que el header X-Session-Id esté presente.
    Si no existe, lanza una excepción HTTP 400.
    """
    session_id = request.headers.get('X-Session-Id')
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Header 'X-Session-Id' is required"
        )
    return session_id


@router.post('/text-to-text')
async def chat_stream(request: Request, q: str = Form(..., description='Pregunta para el modelo')):
    """
    Endpoint para chatear con el agente en modo streaming.
    """
    session_id = get_session_id(request)

    try:
        return StreamingResponse(
            services.process_message_stream(q, session_id=session_id),
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type='text/plain'
        )


@router.get('/history')
async def get_chat_history(request: Request):
    """
    Endpoint para obtener el historial de mensajes de una sesión directamente desde Redis.
    Útil para restaurar el historial después de recargar la página.
    Solo devuelve mensajes de tipo 'human' y 'ai'.
    """
    session_id = get_session_id(request)

    try:
        messages = services.get_history(session_id)
        return {'data': messages}
    except Exception as e:
        logger.error(f'Error getting history: {str(e)}')
        return Response(
            content=f'Error: {str(e)}',
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type='application/json'
        )


@router.post('/clear-history')
async def clear_chat_history(request: Request):
    """
    Endpoint para limpiar el historial de chat de una sesión.
    """
    session_id = get_session_id(request)

    try:
        services.clear_history(session_id)
        return {'status': 'success', 'message': f'History cleared for session {session_id}'}
    except Exception as e:
        logger.error(f'Error clearing history: {str(e)}')
        return Response(
            content=f'Error: {str(e)}',
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type='application/json'
        )
