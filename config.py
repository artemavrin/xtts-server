import os
import torch
from pathlib import Path
from transformers import GenerationConfig

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
OPTIMIZED_GENERATION_CONFIG = GenerationConfig(
    do_sample=False,
    num_beams=1,
    do_stream=True,
    temperature=1.0,  # Более низкая температура для более быстрых, детерминированных выводов
    repetition_penalty=1.0,  # Стандартное значение, регулируйте при необходимости
    pad_token_id=0,  # Добавляем pad_token_id
    bos_token_id=1,  # Добавляем bos_token_id
    eos_token_id=2,  # Добавляем eos_token_id
    return_dict_in_generate=True,  # Добавляем return_dict_in_generate
    max_new_tokens=2048,  # Ограничиваем максимальное количество токенов
    min_new_tokens=1,  # Минимальное количество новых токенов
    length_penalty=1.0,  # Штраф за длину
    no_repeat_ngram_size=3,  # Предотвращение повторения n-грамм
    early_stopping=True  # Ранняя остановка генерации
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
        # Оптимизации для CPU
        torch.set_num_threads(NUM_THREADS)
        return {
            "use_fp16": False,
            "batch_size": 1,
            "use_cudnn": False,
            "use_tf32": False,
            "use_autocast": False
        }

# Применяем оптимизации
model_opts = configure_model_optimizations()

# Настройки потоковой передачи
STREAM_CHUNK_SIZE = int(os.environ.get("STREAM_CHUNK_SIZE", 20))
ADD_WAV_HEADER = os.environ.get("ADD_WAV_HEADER", "True").lower() in ("true", "1", "t")

# Настройки предобработки текста
MAX_TEXT_CHUNK_SIZE = int(os.environ.get("MAX_TEXT_CHUNK_SIZE", 150))
MIN_TEXT_CHUNK_SIZE = int(os.environ.get("MIN_TEXT_CHUNK_SIZE", 50))

# Настройки аудио
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", 24000))
SAMPLE_WIDTH = int(os.environ.get("SAMPLE_WIDTH", 2))
CHANNELS = int(os.environ.get("CHANNELS", 1))