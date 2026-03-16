from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import Response
import os

from ..shared.logger import get_logger
from ..shared.env import env

logger = get_logger(__name__)
router = APIRouter(prefix='/media', tags=['media'])


@router.get('/{filename:path}')
@router.head('/{filename:path}')
async def serve_media_file(filename: str, request: Request):
    """
    Endpoint para servir archivos mediante X-Accel-Redirect.
    """

    media_header = request.headers.get('X-Access-Media')  # example

    if media_header is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

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
