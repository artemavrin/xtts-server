import asyncio
import uuid
import hashlib
import os
import tempfile
import torch
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from fastapi import UploadFile, HTTPException

from models.voice_storage import VoiceStorage
from services.cache import SpeakerCache
from services.audio import ensure_tensor_dimensions
from config import device

# Глобальный словарь для отслеживания активных сессий
active_sessions: Dict[str, Dict[str, Any]] = {}

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
    # 1. Если указан speaker_id, сначала проверяем кэш
    if speaker_id:
        # Проверяем кэш
        cache_key = f"speaker_id:{speaker_id}"
        cached_data = speaker_cache.get(cache_key)
        if cached_data:
            gpt_cond_latent, speaker_embedding = cached_data
            # Проверяем и исправляем размерность gpt_cond_latent
            gpt_cond_latent = ensure_tensor_dimensions(gpt_cond_latent)
            return gpt_cond_latent, speaker_embedding
        
        # 2. Если в кэше нет, проверяем сохраненные голоса
        if voice_storage is not None:
            saved_voice = voice_storage.get_voice(speaker_id)
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
                
                # Обновляем кэш для будущих запросов
                speaker_cache.set(cache_key, (gpt_cond_latent, speaker_embedding))
                
                return gpt_cond_latent, speaker_embedding
        
        # 3. Проверяем встроенные голоса модели
        if hasattr(model, "speaker_manager") and hasattr(model.speaker_manager, "speakers") and speaker_id in model.speaker_manager.speakers:
            speaker_data = model.speaker_manager.speakers[speaker_id]
            gpt_cond_latent = speaker_data["gpt_cond_latent"]
            speaker_embedding = speaker_data["speaker_embedding"]
            
            # Проверяем и исправляем размерность gpt_cond_latent
            gpt_cond_latent = ensure_tensor_dimensions(gpt_cond_latent)
            
            result = (gpt_cond_latent, speaker_embedding)
            # Обновляем кэш для будущих запросов
            speaker_cache.set(cache_key, result)
            
            return result
        
        # Если голос не найден ни в одном из источников
        raise HTTPException(
            status_code=404, detail=f"Speaker ID {speaker_id} not found in cache, saved voices, or model"
        )

    # 4. Для загруженного аудио-файла (UploadFile) через API
    elif audio_file:
        try:
            # Создаем временный файл для сохранения загруженного аудио
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                # Читаем и записываем содержимое во временный файл
                content = await audio_file.read()
                temp_file.write(content)
            
            # Создаем хеш содержимого для кэширования
            file_hash = hashlib.md5(content).hexdigest()
            cache_key = f"file:{file_hash}"
            
            # Проверяем кэш
            cached_data = speaker_cache.get(cache_key)
            if cached_data:
                # Удаляем временный файл, если данные уже в кэше
                os.unlink(temp_path)
                gpt_cond_latent, speaker_embedding = cached_data
                # Проверяем и исправляем размерность gpt_cond_latent
                gpt_cond_latent = ensure_tensor_dimensions(gpt_cond_latent)
                return gpt_cond_latent, speaker_embedding
                
            # Если нет в кэше, обрабатываем файл
            with torch.inference_mode():
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                    temp_path)
            
            # Удаляем временный файл после обработки
            os.unlink(temp_path)
            
            # Проверяем и исправляем размерность gpt_cond_latent
            gpt_cond_latent = ensure_tensor_dimensions(gpt_cond_latent)
            
            # Сохраняем результат в кэш
            result = (gpt_cond_latent, speaker_embedding)
            speaker_cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            # В случае ошибки убеждаемся, что временный файл удален
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise HTTPException(
                status_code=400, detail=f"Error processing audio file: {str(e)}")

    # 5. Для аудиофайлов по пути на сервере
    elif audio_file_path and os.path.exists(audio_file_path):
        try:
            # Создаем хеш файла для кэширования
            file_hash = None
            with open(audio_file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            cache_key = f"file:{file_hash}"
            
            # Проверяем кэш
            cached_data = speaker_cache.get(cache_key)
            if cached_data:
                gpt_cond_latent, speaker_embedding = cached_data
                # Проверяем и исправляем размерность gpt_cond_latent
                gpt_cond_latent = ensure_tensor_dimensions(gpt_cond_latent)
                return gpt_cond_latent, speaker_embedding
                
            # Если нет в кэше, обрабатываем файл
            with torch.inference_mode():
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                    audio_file_path)
            
            # Проверяем и исправляем размерность gpt_cond_latent
            gpt_cond_latent = ensure_tensor_dimensions(gpt_cond_latent)
            
            # Сохраняем результат в кэш
            result = (gpt_cond_latent, speaker_embedding)
            speaker_cache.set(cache_key, result)
            
            return result
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error processing audio file: {str(e)}")

    else:
        raise HTTPException(
            status_code=400,
            detail="No speaker provided. Please provide either a speaker_id, upload an audio file, or specify a valid audio_file_path."
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