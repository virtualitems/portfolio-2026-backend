import faiss
import os

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from server.shared.logger import get_logger
from server.persons.routes import router as persons_router
from server.reports.routes import router as reports_router
from server.chatbot.routes import router as chatbot_router
from server.media.routes import router as media_router
from server.vision.routes import router as vision_router

logger = get_logger(__name__)

app = FastAPI()

gpu_available = faiss.get_num_gpus()
cpu_available = os.cpu_count() or 1

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS'],
    allow_headers=['*'],
)

app.include_router(media_router)
app.include_router(persons_router, prefix='/api')
app.include_router(reports_router, prefix='/api')
app.include_router(chatbot_router, prefix='/api')
app.include_router(vision_router, prefix='/ws')

@app.get('/api/health', status_code=status.HTTP_204_NO_CONTENT)
async def health_check():
    """Endpoint to check the health of the server"""
