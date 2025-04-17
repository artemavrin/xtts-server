import os
import torch
from pathlib import Path

# Настройки окружения
DEBUG = os.environ.get("DEBUG", "False").lower() in ("true", "1", "t")
PORT = int(os.environ.get("PORT", 8000))
NUM_THREADS = int(os.environ.get("NUM_THREADS", os.cpu_count()))
MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", 10))
CUSTOM_MODEL_PATH = os.environ.get("CUSTOM_MODEL_PATH", "/app/tts_models")
USE_CPU = os.environ.get("USE_CPU", "0") == "1"

# Настройки кэша
SPEAKER_CACHE_TTL = int(os.environ.get("SPEAKER_CACHE_TTL", 3600))  # 1 час по умолчанию
SPEAKER_CACHE_MAX_SIZE = int(os.environ.get("SPEAKER_CACHE_MAX_SIZE", 100))

# Настройки предзагрузки голосов
PRELOAD_VOICES = os.environ.get("PRELOAD_VOICES", "").split(",")
MAX_PRELOAD_VOICES = int(os.environ.get("MAX_PRELOAD_VOICES", 10))

# Настройки хранилища голосов
VOICES_DIR = Path(os.environ.get("VOICES_DIR", "./saved_voices"))
VOICES_DIR.mkdir(exist_ok=True)

# Определяем устройство (CPU/GPU)
device = torch.device("cuda" if not USE_CPU and torch.cuda.is_available() else "cpu")

# Оптимизированные параметры генерации для более быстрого вывода
from transformers import GenerationConfig

OPTIMIZED_GENERATION_CONFIG = GenerationConfig(
    do_sample=False,
    num_beams=1,
    do_stream=True,
    temperature=1.0,  # Более низкая температура для более быстрых, детерминированных выводов
    max_new_tokens=250,  # Ограничение максимального количества токенов на чанк
    repetition_penalty=1.0  # Стандартное значение, регулируйте при необходимости
)

# Функция для применения оптимизаций в зависимости от доступного оборудования
def configure_model_optimizations():
    """Применяем оптимизации производительности в зависимости от доступного оборудования"""
    if device.type == "cuda":
        # Оптимизации для GPU
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        
        # Если у вас достаточно видеопамяти, используйте эти настройки:
        if torch.cuda.get_device_properties(0).total_memory > 8 * 1024 * 1024 * 1024:  # > 8GB
            return {
                "use_fp16": True,
                "batch_size": 2,  # Можно увеличить в зависимости от объема видеопамяти
            }
        else:
            return {
                "use_fp16": True,
                "batch_size": 1,
            }
    else:
        # Оптимизации для CPU
        torch.set_num_threads(NUM_THREADS)
        return {
            "use_fp16": False,
            "batch_size": 1,
        }