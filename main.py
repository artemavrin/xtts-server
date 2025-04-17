import asyncio
import traceback
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import (
    MAX_CONCURRENT_REQUESTS, VOICES_DIR, SPEAKER_CACHE_TTL, 
    SPEAKER_CACHE_MAX_SIZE, PRELOAD_VOICES, MAX_PRELOAD_VOICES
)
from services.tts import TTSManager
from models.voice_storage import VoiceStorage
from services.cache import SpeakerCache
from services.utils import preload_voices_to_cache
from api.routes import router

# Инициализация приложения
app = FastAPI(
    title="XTTS Streaming Server",
    description="""XTTS Streaming server with support for concurrent requests and voice cloning""",
    version="0.2.0",
    docs_url="/",
)

# Глобальные объекты
tts_manager = TTSManager()
voice_storage = VoiceStorage(VOICES_DIR)
speaker_cache = SpeakerCache(ttl=SPEAKER_CACHE_TTL, max_size=SPEAKER_CACHE_MAX_SIZE)

# Добавление CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Обработчик исключений
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Глобальный обработчик исключений"""
    print(f"Global exception handler caught: {str(exc)}", flush=True)
    print(f"Exception traceback: {traceback.format_exc()}", flush=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "detail": str(exc),
            "type": exc.__class__.__name__
        }
    )

# Функция запуска сервера
@app.on_event("startup")
async def startup_event():
    """
    Выполняется при запуске сервера - инициализация всех компонентов
    """
    global tts_manager, voice_storage, speaker_cache
    
    try:
        # Инициализация TTS модели
        tts_manager.initialize_model()
        
        # Создание семафора для контроля доступа к модели
        tts_manager.set_semaphore(MAX_CONCURRENT_REQUESTS)
        
        # Предзагрузка голосов в кэш
        await preload_voices_to_cache(
            tts_manager.model,
            speaker_cache,
            voice_storage,
            PRELOAD_VOICES,
            MAX_PRELOAD_VOICES
        )
        
        print("Server initialized and ready to handle requests", flush=True)
    except Exception as e:
        print(f"Error during startup: {str(e)}", flush=True)
        print(f"Startup error traceback: {traceback.format_exc()}", flush=True)
        raise

# Регистрация маршрутов
app.include_router(router)

# Если файл запускается напрямую
if __name__ == "__main__":
    import uvicorn
    import os
    
    # Получаем порт из переменных окружения или используем порт по умолчанию
    port = int(os.environ.get("PORT", 8000))
    
    # Запускаем сервер
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=1  # Несколько воркеров не рекомендуется с GPU-моделями
    )