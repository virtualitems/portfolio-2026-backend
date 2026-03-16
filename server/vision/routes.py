from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..shared.logger import get_logger
from .safety_detector import safety_detector

logger = get_logger(__name__)
router = APIRouter(prefix='/vision', tags=['vision'])


@router.websocket('/stream-safety')
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
