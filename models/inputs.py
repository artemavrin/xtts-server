from typing import List, Optional
from enum import Enum
from pydantic import BaseModel

class TTSInputs(BaseModel):
    """Модель для входных данных синтеза речи"""
    text: str
    language: str
    speaker_id: Optional[str] = None
    audio_file_path: Optional[str] = None
    gpt_cond_latent: Optional[List[float]] = None  
    speaker_embedding: Optional[List[float]] = None

class StreamingOptions(BaseModel):
    """Настройки потоковой передачи аудио"""
    stream_chunk_size: int = 20
    add_wav_header: bool = True

class CloneResponseFormat(str, Enum):
    """Формат ответа для клонирования голоса"""
    json = "json"           # Возвращает данные в JSON
    save = "save"           # Сохраняет голос в хранилище
    json_and_save = "both"  # И возвращает JSON, и сохраняет

class CloneVoiceOptions(BaseModel):
    """Опции для клонирования голоса"""
    save_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    response_format: CloneResponseFormat = CloneResponseFormat.json