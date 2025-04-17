import asyncio
import uuid
import hashlib
import os
import tempfile
import torch
import numpy as np
import re
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from fastapi import UploadFile, HTTPException

from models.voice_storage import VoiceStorage
from services.cache import SpeakerCache
from services.audio import ensure_tensor_dimensions, ensure_speaker_embedding_dimensions
from config import (
    device,
    MAX_TEXT_CHUNK_SIZE,
    MIN_TEXT_CHUNK_SIZE,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    CHANNELS
)

# Глобальный словарь для отслеживания активных сессий
active_sessions: Dict[str, Dict[str, Any]] = {}

def preprocess_text(text: str) -> List[str]:
    """
    Предобработка текста для более эффективной потоковой передачи
    
    Разделение длинных параграфов на меньшие, управляемые фрагменты, которые
    могут быть более эффективно обработаны TTS-моделью.
    
    Args:
        text: Исходный текст
        
    Returns:
        Список текстовых фрагментов
    """
    # Разделение текста по границам предложений
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Группировка предложений в фрагменты разумного размера
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Если добавление этого предложения сделает фрагмент слишком длинным, начинаем новый фрагмент
        if len(current_chunk) + len(sentence) > MAX_TEXT_CHUNK_SIZE:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Добавляем последний фрагмент, если он не пустой
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def generate_silence(duration_ms: int = 100, sample_rate: int = SAMPLE_RATE) -> bytes:
    """
    Генерирует данные тишины для более быстрого начального ответа
    
    Args:
        duration_ms: Длительность тишины в миллисекундах
        sample_rate: Частота дискретизации
        
    Returns:
        Байты аудио-данных тишины
    """
    num_samples = int(duration_ms * sample_rate / 1000)
    silence = np.zeros(num_samples, dtype=np.int16)
    return silence.tobytes()

async def _cleanup_session(sid: str, delay: int = 60) -> None:
    """
    Очистка сессии после задержки
    
    Args:
        sid: ID сессии
        delay: Задержка перед очисткой в секундах
    """
    await asyncio.sleep(delay)
    if sid in active_sessions:
        del active_sessions[sid]
    print(f"Cleaned up session {sid}", flush=True)

def generate_session_id() -> str:
    """
    Генерирует уникальный ID сессии
    
    Returns:
        Уникальный ID сессии
    """
    return str(uuid.uuid4())

async def get_speaker_data(
    model: Any,
    speaker_cache: SpeakerCache,
    voice_storage: Optional[VoiceStorage] = None,
    speaker_id: Optional[str] = None,
    audio_file: Optional[UploadFile] = None,
    audio_file_path: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Получение данных голоса из разных источников с оптимизацией через кэширование
    
    Args:
        model: Модель TTS
        speaker_cache: Кэш голосов
        voice_storage: Хранилище голосов (опционально)
        speaker_id: ID голоса
        audio_file: Загруженный аудиофайл
        audio_file_path: Путь к аудиофайлу на сервере
        
    Returns:
        Кортеж (gpt_cond_latent, speaker_embedding)
        
    Raises:
        HTTPException: Если голос не найден или произошла ошибка при обработке
    """
    try:
        # Проверяем кэш
        if speaker_id:
            cached_data = speaker_cache.get(speaker_id)
            if cached_data is not None:
                print(f"Using cached voice data for speaker_id: {speaker_id}", flush=True)
                return cached_data
        
        # Получаем данные голоса
        if speaker_id:
            if voice_storage and voice_storage.has_voice(speaker_id):
                print(f"Loading voice data from storage for speaker_id: {speaker_id}", flush=True)
                voice_data = voice_storage.get_voice(speaker_id)
                if voice_data is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Voice data not found for speaker_id: {speaker_id}"
                    )
                # Преобразуем данные в тензоры с правильным dtype
                gpt_cond_latent = torch.tensor(voice_data["gpt_cond_latent"], dtype=torch.float).to(model.device)
                speaker_embedding = torch.tensor(voice_data["speaker_embedding"], dtype=torch.float).to(model.device)
                
                # Восстанавливаем форму тензора gpt_cond_latent
                if "gpt_cond_latent_shape" in voice_data:
                    gpt_cond_latent = gpt_cond_latent.reshape(*voice_data["gpt_cond_latent_shape"])
            else:
                print(f"Using built-in voice for speaker_id: {speaker_id}", flush=True)
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_id)
        elif audio_file or audio_file_path:
            audio_path = audio_file_path if audio_file_path else await save_upload_file(audio_file)
            print(f"Generating voice data from audio file: {audio_path}", flush=True)
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=audio_path)
        else:
            raise HTTPException(
                status_code=400,
                detail="Either speaker_id or audio_file must be provided"
            )
        
        # Исправляем размерность speaker_embedding (гарантирует минимум 2D: [1, 512])
        speaker_embedding = ensure_speaker_embedding_dimensions(speaker_embedding)
        
        # Добавляем последнюю размерность, ТОЛЬКО если она отсутствует (для Conv1d)
        if speaker_embedding.dim() == 2:
            speaker_embedding = speaker_embedding.unsqueeze(-1) # [1, 512] -> [1, 512, 1]
        # Если уже 3D ([1, 512, 1]), ничего не делаем
        
        # Кэшируем результат (с формой [1, 512, 1])
        if speaker_id:
            speaker_cache.set(speaker_id, (gpt_cond_latent, speaker_embedding))
        
        return gpt_cond_latent, speaker_embedding
        
    except Exception as e:
        print(f"Error in get_speaker_data: {str(e)}", flush=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing voice data: {str(e)}"
        )

async def make_async_generator(sync_generator):
    """
    Преобразует синхронный генератор в асинхронный генератор
    
    Args:
        sync_generator: Синхронный генератор или итератор
        
    Yields:
        Элементы из синхронного генератора
    """
    try:
        for item in sync_generator:
            yield item
    except Exception as e:
        print(f"Error in async generator wrapper: {str(e)}", flush=True)
        raise

async def preload_voices_to_cache(
    model: Any,
    speaker_cache: SpeakerCache,
    voice_storage: Optional[VoiceStorage] = None,
    voices_to_preload: List[str] = None,
    max_preload: int = 10
) -> None:
    """
    Предзагружает часто используемые голоса в кэш для быстрого доступа
    
    Args:
        model: Модель TTS
        speaker_cache: Кэш голосов
        voice_storage: Хранилище голосов
        voices_to_preload: Список голосов для предзагрузки
        max_preload: Максимальное количество предзагружаемых голосов
    """
    print("Preloading voices to cache...", flush=True)
    
    # Список голосов для предзагрузки
    if voices_to_preload is None:
        voices_to_preload = []
    
    # Добавляем все голоса из модели, если они есть
    if hasattr(model, "speaker_manager") and hasattr(model.speaker_manager, "speakers"):
        voices_to_preload.extend(list(model.speaker_manager.speakers.keys()))
    
    # Добавляем сохраненные голоса, если хранилище инициализировано
    if voice_storage is not None:
        saved_voices = voice_storage.list_voices()
        voices_to_preload.extend(list(saved_voices.keys()))
    
    # Удаляем дубликаты
    voices_to_preload = list(set(voices_to_preload))
    
    # Ограничиваем количество предзагружаемых голосов
    voices_to_preload = voices_to_preload[:max_preload]
    
    if not voices_to_preload:
        print("No voices to preload.", flush=True)
        return
    
    # Загружаем голоса в кэш
    loaded_count = 0
    for voice_id in voices_to_preload:
        try:
            # Пытаемся загрузить из разных источников
            
            # 1. Проверяем сохраненные голоса
            if voice_storage is not None:
                saved_voice = voice_storage.get_voice(voice_id)
                if saved_voice:
                    # Преобразуем данные из JSON в тензоры
                    gpt_cond_latent = torch.tensor(saved_voice["gpt_cond_latent"]).to(device)
                    speaker_embedding = torch.tensor(saved_voice["speaker_embedding"]).to(device)
                    
                    # Если есть информация о форме тензора, используем её
                    if "gpt_cond_latent_shape" in saved_voice:
                        tensor_shape = saved_voice["gpt_cond_latent_shape"]
                        gpt_cond_latent = gpt_cond_latent.reshape(*tensor_shape)
                    
                    # Проверяем и исправляем размерность gpt_cond_latent
                    gpt_cond_latent = ensure_tensor_dimensions(gpt_cond_latent)
                    
                    # Добавляем в кэш
                    speaker_cache.set(f"speaker_id:{voice_id}", (gpt_cond_latent, speaker_embedding))
                    loaded_count += 1
                    continue
            
            # 2. Проверяем встроенные голоса модели
            if hasattr(model, "speaker_manager") and voice_id in model.speaker_manager.speakers:
                speaker_data = model.speaker_manager.speakers[voice_id]
                gpt_cond_latent = speaker_data["gpt_cond_latent"]
                speaker_embedding = speaker_data["speaker_embedding"]
                
                # Проверяем и исправляем размерность gpt_cond_latent
                gpt_cond_latent = ensure_tensor_dimensions(gpt_cond_latent)
                
                # Добавляем в кэш
                speaker_cache.set(f"speaker_id:{voice_id}", (gpt_cond_latent, speaker_embedding))
                loaded_count += 1
        
        except Exception as e:
            print(f"Error preloading voice {voice_id}: {str(e)}", flush=True)
    
    print(f"Preloaded {loaded_count}/{len(voices_to_preload)} voices to cache.", flush=True)

async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Сохраняет загруженный файл во временную директорию
    
    Args:
        upload_file: Загруженный файл
        
    Returns:
        Путь к сохраненному файлу
    """
    try:
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = temp_file.name
            
            # Читаем и записываем содержимое файла
            content = await upload_file.read()
            temp_file.write(content)
            
            return temp_path
    except Exception as e:
        print(f"Error saving upload file: {str(e)}", flush=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error saving upload file: {str(e)}"
        )