import asyncio
import traceback
import torch
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import GenerationConfig

from config import (
    MAX_CONCURRENT_REQUESTS, VOICES_DIR, SPEAKER_CACHE_TTL, 
    SPEAKER_CACHE_MAX_SIZE, PRELOAD_VOICES, MAX_PRELOAD_VOICES
)
from services.tts import TTSManager
from models.voice_storage import VoiceStorage
from services.cache import SpeakerCache
from services.utils import preload_voices_to_cache
from api.routes import router

# Optimized generation parameters for faster output
OPTIMIZED_GENERATION_CONFIG = GenerationConfig(
    do_sample=False,
    num_beams=1,
    do_stream=True,
    temperature=1.0,  # Lower temperature for faster, more deterministic outputs
    repetition_penalty=1.0  # Standard value, adjust as needed
)

# Multithreading and device setup
torch.set_num_threads(int(os.environ.get("NUM_THREADS", os.cpu_count())))
device = torch.device("cuda" if os.environ.get(
    "USE_CPU", "0") == "0" and torch.cuda.is_available() else "cpu")

# Function to apply optimizations based on available hardware
def configure_model_optimizations():
    """Apply performance optimizations based on available hardware"""
    if device.type == "cuda":
        # GPU Optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

        # If you have enough VRAM, use these settings:
        if torch.cuda.get_device_properties(0).total_memory > 8 * 1024 * 1024 * 1024:  # > 8GB
            return {
                "use_fp16": True,
                "batch_size": 2,  # Can be increased depending on VRAM amount
            }
        else:
            return {
                "use_fp16": True,
                "batch_size": 1,
            }
    else:
        # CPU Optimizations
        torch.set_num_threads(int(os.environ.get("NUM_THREADS", os.cpu_count())))
        return {
            "use_fp16": False,
            "batch_size": 1,
        }

# Apply optimizations
model_opts = configure_model_optimizations()

# Application initialization
app = FastAPI(
    title="XTTS Streaming Server",
    description="""XTTS Streaming server with support for concurrent requests and voice cloning""",
    version="0.2.0",
    docs_url="/",
)

# Global objects
tts_manager = TTSManager()
voice_storage = VoiceStorage(VOICES_DIR)
speaker_cache = SpeakerCache(ttl=SPEAKER_CACHE_TTL, max_size=SPEAKER_CACHE_MAX_SIZE)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    print(f"Global exception handler caught: {str(exc)}", flush=True)
    print(f"Exception traceback: {traceback.format_exc()}", flush=True)

    # More detailed error handling
    error_detail = str(exc)
    if "Error processing audio file" in error_detail:
        status_code = 400
    elif "Speaker ID not found" in error_detail:
        status_code = 404
    else:
        status_code = 500
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error": True,
            "detail": error_detail,
            "type": exc.__class__.__name__
        }
    )

# Server startup function
@app.on_event("startup")
async def startup_event():
    """
    Executed on server startup - initializes all components
    """
    global tts_manager, voice_storage, speaker_cache
    
    try:
        # Initialize TTS model
        tts_manager.initialize_model()
        
        # Create semaphore to control access to the model
        tts_manager.set_semaphore(MAX_CONCURRENT_REQUESTS)
        
        # Preload voices into cache
        await preload_voices_to_cache(
            tts_manager.model,
            speaker_cache,
            voice_storage,
            PRELOAD_VOICES,
            MAX_PRELOAD_VOICES
        )
        
        print("Server initialized and ready to handle requests", flush=True)
        print(f"Using device: {device}", flush=True)
        print(f"Model optimizations: {model_opts}", flush=True)
    except Exception as e:
        print(f"Error during startup: {str(e)}", flush=True)
        print(f"Startup error traceback: {traceback.format_exc()}", flush=True)
        raise

# Register routes
app.include_router(router)

# If the file is run directly
if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variables or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=1  # Multiple workers are not recommended with GPU models
    )