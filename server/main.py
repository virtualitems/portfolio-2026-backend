import faiss
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .shared.logger import get_logger
from .vision.safety_detector import safety_detector
from .chatbot.agent import chatbot_agent
from .shared.env import env
from .shared.redis import create_session_store
from .persons.routes import router as persons_router
from .reports.routes import router as reports_router

logger = get_logger(__name__)

app = FastAPI()

gpu_available = faiss.get_num_gpus()
cpu_available = os.cpu_count() or 1

# Crear instancia del session store para acceso directo a Redis
session_store = create_session_store()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS'],
    allow_headers=['*'],
)

app.include_router(persons_router)
app.include_router(reports_router)

@app.get('/health')
async def health():
    return Response(status_code=200, content=f'OK {gpu_available}G {cpu_available}C')

@app.get('/media/{filename:path}')
@app.head('/media/{filename:path}')
async def serve_media_file(filename: str, request: Request):
    """
    Endpoint para servir archivos mediante X-Accel-Redirect.
    """
    try:
        base_path = env.get('MEDIA_FILES_DIR_PATH')
        file_path = os.path.join(base_path, filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        return Response(
            content='',
            headers={ 'X-Accel-Redirect': f'/mediafiles/{filename}' }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving media file {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/chatbot/text-to-text')
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

@app.get('/chatbot/history')
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

@app.post('/chatbot/clear-history')
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

@app.websocket('/vision/stream-safety')
async def stream_safety_detection(websocket: WebSocket):
    """WebSocket endpoint for real-time safety detection"""
    await websocket.accept()

    try:
        safety_detector.load_model()

        while True:
            data = await websocket.receive_text()

            if not data.startswith('data:image'):
                continue

            processed_frame = safety_detector.process_base64_frame(data, confidence_threshold=0.5)

            if processed_frame is None:
                logger.warning("Failed to process frame, skipping")
                continue

            await websocket.send_text(processed_frame)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Error in safety detection stream: {str(e)}")
        try:
            await websocket.close()
        except:
            pass
