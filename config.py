import os
import torch
from pathlib import Path
from transformers import GenerationConfig

# Environment settings
DEBUG = os.environ.get("DEBUG", "False").lower() in ("true", "1", "t")
PORT = int(os.environ.get("PORT", 8000))
NUM_THREADS = int(os.environ.get("NUM_THREADS", os.cpu_count()))
MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", 10))
CUSTOM_MODEL_PATH = os.environ.get("CUSTOM_MODEL_PATH", "/app/tts_models")
USE_CPU = os.environ.get("USE_CPU", "0") == "1"

# Cache settings
SPEAKER_CACHE_TTL = int(os.environ.get("SPEAKER_CACHE_TTL", 3600))  # 1 hour by default
SPEAKER_CACHE_MAX_SIZE = int(os.environ.get("SPEAKER_CACHE_MAX_SIZE", 100))

# Voice preloading settings
PRELOAD_VOICES = os.environ.get("PRELOAD_VOICES", "").split(",")
MAX_PRELOAD_VOICES = int(os.environ.get("MAX_PRELOAD_VOICES", 10))

# Voice storage settings
VOICES_DIR = Path(os.environ.get("VOICES_DIR", "./saved_voices"))
VOICES_DIR.mkdir(exist_ok=True)

# Define device (CPU/GPU)
device = torch.device("cuda" if not USE_CPU and torch.cuda.is_available() else "cpu")

# Optimized generation parameters for faster output
OPTIMIZED_GENERATION_CONFIG = GenerationConfig(
    do_sample=False,
    num_beams=1,
    do_stream=True,
    temperature=1.0,  # Lower temperature for faster, more deterministic outputs
    repetition_penalty=1.0,  # Standard value, adjust as needed
    pad_token_id=0,  # Add pad_token_id
    bos_token_id=1,  # Add bos_token_id
    eos_token_id=2,  # Add eos_token_id
    return_dict_in_generate=True,  # Add return_dict_in_generate
    max_new_tokens=2048,  # Limit maximum number of tokens
    min_new_tokens=1,  # Minimum number of new tokens
    length_penalty=1.0,  # Length penalty
    no_repeat_ngram_size=3,  # Prevent n-gram repetition
    early_stopping=True  # Early generation stopping
)

# Function to apply optimizations based on available hardware
def configure_model_optimizations():
    """Apply performance optimizations based on available hardware"""
    if device.type == "cuda":
        # GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        
        # If you have enough VRAM, use these settings:
        if torch.cuda.get_device_properties(0).total_memory > 8 * 1024 * 1024 * 1024:  # > 8GB
            return {
                "use_fp16": True,
                "batch_size": 2,  # Can be increased depending on VRAM amount
                "use_cudnn": True,
                "use_tf32": True,
                "use_autocast": True
            }
        else:
            return {
                "use_fp16": True,
                "batch_size": 1,
                "use_cudnn": True,
                "use_tf32": True,
                "use_autocast": False
            }
    else:
        # CPU optimizations
        torch.set_num_threads(NUM_THREADS)
        return {
            "use_fp16": False,
            "batch_size": 1,
            "use_cudnn": False,
            "use_tf32": False,
            "use_autocast": False
        }

# Apply optimizations
model_opts = configure_model_optimizations()

# Streaming settings
STREAM_CHUNK_SIZE = int(os.environ.get("STREAM_CHUNK_SIZE", 20))
ADD_WAV_HEADER = os.environ.get("ADD_WAV_HEADER", "True").lower() in ("true", "1", "t")

# Text preprocessing settings
MAX_TEXT_CHUNK_SIZE = int(os.environ.get("MAX_TEXT_CHUNK_SIZE", 150))
MIN_TEXT_CHUNK_SIZE = int(os.environ.get("MIN_TEXT_CHUNK_SIZE", 50))

# Audio settings
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", 24000))
SAMPLE_WIDTH = int(os.environ.get("SAMPLE_WIDTH", 2))
CHANNELS = int(os.environ.get("CHANNELS", 1))