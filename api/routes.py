import torch
import uuid
from fastapi import APIRouter, UploadFile, HTTPException, Depends
from fastapi.responses import StreamingResponse

from models.inputs import TTSInputs, StreamingOptions, CloneVoiceOptions
from models.voice_storage import VoiceStorage
from services.cache import SpeakerCache
from services.tts import TTSManager
from services.utils import get_speaker_data, generate_session_id, active_sessions
from services.audio import ensure_tensor_dimensions

router = APIRouter()

# Зависимости для инъекции сервисов
def get_tts_manager():
    """Получение менеджера TTS"""
    from main import tts_manager
    return tts_manager

def get_voice_storage():
    """Получение хранилища голосов"""
    from main import voice_storage
    return voice_storage

def get_speaker_cache():
    """Получение кэша голосов"""
    from main import speaker_cache
    return speaker_cache

@router.post("/tts_stream")
async def tts_stream(
    request: TTSInputs, 
    options: StreamingOptions = StreamingOptions(),
    tts_manager: TTSManager = Depends(get_tts_manager),
    voice_storage: VoiceStorage = Depends(get_voice_storage),
    speaker_cache: SpeakerCache = Depends(get_speaker_cache)
):
    """
    Потоковая передача текста в речь
    
    Поддерживает несколько способов указания голоса:
    1. По ID сохраненного голоса (speaker_id)
    2. По встроенному ID голоса модели (speaker_id)
    3. По загруженному аудиофайлу через API (audio_file_path)
    4. По прямой передаче параметров голоса (gpt_cond_latent, speaker_embedding)
    """
    # Генерируем ID сессии
    session_id = generate_session_id()
    
    # Инициализируем сессию
    active_sessions[session_id] = {
        "status": "initializing", 
        "text": request.text,
        "language": request.language,
        "speaker_id": request.speaker_id,
        "created_at": 0  # Будет обновлено при старте генерации
    }
    
    print(f"[{session_id}] TTS Stream requested for text: '{request.text[:50]}...' with language: {request.language}", flush=True)
    
    async def stream_generator():
        try:
            # Проверяем, были ли переданы параметры голоса напрямую
            if request.gpt_cond_latent is not None and request.speaker_embedding is not None:
                print(f"[{session_id}] Using provided voice embeddings", flush=True)
                # Преобразуем списки в тензоры
                gpt_cond_latent = torch.tensor(request.gpt_cond_latent).to(tts_manager.model.device)
                speaker_embedding = torch.tensor(request.speaker_embedding).to(tts_manager.model.device)
                
                # Проверяем и исправляем размерность
                original_shape = gpt_cond_latent.shape
                gpt_cond_latent = ensure_tensor_dimensions(gpt_cond_latent)
                new_shape = gpt_cond_latent.shape
                
                print(f"[{session_id}] Voice tensor shape: {original_shape} -> {new_shape}", flush=True)
            else:
                # Получаем данные диктора из других источников
                print(f"[{session_id}] Fetching voice data for speaker_id: {request.speaker_id}", flush=True)
                gpt_cond_latent, speaker_embedding = await get_speaker_data(
                    tts_manager.model,
                    speaker_cache,
                    voice_storage,
                    speaker_id=request.speaker_id,
                    audio_file_path=request.audio_file_path
                )
                
                # Логируем форму тензоров
                print(f"[{session_id}] Voice data fetched. gpt_cond_latent shape: {gpt_cond_latent.shape}, "
                     f"speaker_embedding shape: {speaker_embedding.shape}", flush=True)
            
            # Используем процессор текста по чанкам для длинных текстов или обычный генератор для коротких
            if len(request.text) > 150:
                print(f"[{session_id}] Using chunked processing for long text", flush=True)
                async for chunk in tts_manager.process_text_in_chunks(
                    session_id,
                    request.text,
                    request.language,
                    gpt_cond_latent,
                    speaker_embedding,
                    options.stream_chunk_size,
                    options.add_wav_header
                ):
                    yield chunk
            else:
                print(f"[{session_id}] Using direct streaming for short text", flush=True)
                async for chunk in tts_manager.generate_stream(
                    session_id,
                    request.text,
                    request.language,
                    gpt_cond_latent,
                    speaker_embedding,
                    options.stream_chunk_size,
                    options.add_wav_header
                ):
                    yield chunk
                
        except Exception as e:
            # Обрабатываем ошибки
            if session_id in active_sessions:
                active_sessions[session_id]["status"] = "error"
                active_sessions[session_id]["error"] = str(e)
            
            # Подробное логирование ошибки
            print(f"[{session_id}] Error in stream_generator: {str(e)}", flush=True)
            import traceback
            print(f"[{session_id}] Error traceback: {traceback.format_exc()}", flush=True)
            
            raise
    
    # Возвращаем потоковый ответ с асинхронным генератором
    return StreamingResponse(
        stream_generator(),
        media_type="audio/wav",
        headers={"X-Session-ID": session_id}
    )

@router.post("/tts")
async def tts(
    request: TTSInputs,
    tts_manager: TTSManager = Depends(get_tts_manager),
    voice_storage: VoiceStorage = Depends(get_voice_storage),
    speaker_cache: SpeakerCache = Depends(get_speaker_cache)
):
    """
    Генерирует полное аудио (не в потоковом режиме)
    
    Поддерживает несколько способов указания голоса:
    1. По ID сохраненного голоса (speaker_id)
    2. По встроенному ID голоса модели (speaker_id)
    3. По загруженному аудиофайлу через API (audio_file_path)
    4. По прямой передаче параметров голоса (gpt_cond_latent, speaker_embedding)
    """
    try:
        # Проверяем, были ли переданы параметры голоса напрямую
        if request.gpt_cond_latent is not None and request.speaker_embedding is not None:
            # Преобразуем списки в тензоры
            gpt_cond_latent = torch.tensor(request.gpt_cond_latent).to(tts_manager.model.device)
            speaker_embedding = torch.tensor(request.speaker_embedding).to(tts_manager.model.device)
            
            # Проверяем и исправляем размерность
            gpt_cond_latent = ensure_tensor_dimensions(gpt_cond_latent)
        else:
            # Получаем данные диктора из других источников
            gpt_cond_latent, speaker_embedding = await get_speaker_data(
                tts_manager.model,
                speaker_cache,
                voice_storage,
                speaker_id=request.speaker_id,
                audio_file_path=request.audio_file_path
            )

        # Генерируем речь
        audio_base64 = await tts_manager.synthesize_full(
            request.text,
            request.language,
            gpt_cond_latent,
            speaker_embedding
        )
        
        return {"audio": audio_base64}
    
    except Exception as e:
        print(f"Error in tts: {str(e)}", flush=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating speech: {str(e)}"
        )

@router.post("/clone_speaker")
async def clone_speaker(
    wav_file: UploadFile,
    options: CloneVoiceOptions = None,
    tts_manager: TTSManager = Depends(get_tts_manager),
    voice_storage: VoiceStorage = Depends(get_voice_storage),
    speaker_cache: SpeakerCache = Depends(get_speaker_cache)
):
    """
    Клонирование голоса из аудиофайла
    
    Поддерживает различные форматы ответа:
    - json: возвращает данные голоса в JSON
    - save: сохраняет голос в хранилище и возвращает ID
    - both: и сохраняет голос, и возвращает данные в JSON
    """
    try:
        # Если опции не указаны, используем значения по умолчанию
        if options is None:
            options = CloneVoiceOptions()
        
        # Проверяем, что файл - аудио
        if not wav_file.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=400, 
                detail=f"Uploaded file is not an audio file. Content type: {wav_file.content_type}"
            )

        # Получаем данные голоса из аудиофайла
        gpt_cond_latent, speaker_embedding = await get_speaker_data(
            tts_manager.model,
            speaker_cache,
            voice_storage,
            audio_file=wav_file
        )
        
        # Подготавливаем данные голоса для JSON-ответа
        voice_data = {
            "gpt_cond_latent": gpt_cond_latent.cpu().squeeze().half().tolist(),
            "speaker_embedding": speaker_embedding.cpu().squeeze().half().tolist(),
        }
        
        # Обрабатываем различные форматы ответа
        if options.response_format == "json":
            # Просто возвращаем данные голоса
            return voice_data
            
        elif options.response_format == "save":
            # Сохраняем голос в хранилище
            if not options.save_id:
                # Если ID не указан, генерируем случайный
                options.save_id = f"voice-{uuid.uuid4().hex[:8]}"
                
            # Сохраняем голос
            voice_storage.save_voice(
                options.save_id,
                gpt_cond_latent,
                speaker_embedding,
                options.name or options.save_id,
                options.description
            )
            
            return {
                "status": "success",
                "message": f"Voice saved successfully",
                "voice_id": options.save_id
            }
            
        elif options.response_format == "both":
            # И сохраняем, и возвращаем данные
            if not options.save_id:
                # Если ID не указан, генерируем случайный
                options.save_id = f"voice-{uuid.uuid4().hex[:8]}"
                
            # Сохраняем голос
            voice_storage.save_voice(
                options.save_id,
                gpt_cond_latent,
                speaker_embedding,
                options.name or options.save_id,
                options.description
            )
            
            # Возвращаем и данные голоса, и информацию о сохранении
            return {
                "status": "success",
                "message": f"Voice cloned and saved successfully",
                "voice_id": options.save_id,
                "voice_data": voice_data
            }
    
    except Exception as e:
        # Более информативная обработка ошибок
        error_msg = str(e)
        if "Error processing audio file" in error_msg:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process audio: {error_msg}"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Server error: {error_msg}"
            )

@router.get("/saved_voices")
async def list_saved_voices(
    voice_storage: VoiceStorage = Depends(get_voice_storage)
):
    """Возвращает список всех сохраненных голосов"""
    return voice_storage.list_voices()

@router.delete("/saved_voices/{voice_id}")
async def delete_saved_voice(
    voice_id: str,
    voice_storage: VoiceStorage = Depends(get_voice_storage)
):
    """Удаляет сохраненный голос"""
    if voice_storage.delete_voice(voice_id):
        return {"status": "success", "message": f"Voice {voice_id} deleted"}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Voice with ID {voice_id} not found"
        )

@router.post("/save_voice/{voice_id}")
async def save_voice(
    voice_id: str,
    wav_file: UploadFile,
    name: str = None,
    description: str = None,
    tts_manager: TTSManager = Depends(get_tts_manager),
    voice_storage: VoiceStorage = Depends(get_voice_storage),
    speaker_cache: SpeakerCache = Depends(get_speaker_cache)
):
    """Клонирует голос из аудиофайла и сохраняет его для последующего использования"""
    try:
        # Получаем данные голоса из аудиофайла
        gpt_cond_latent, speaker_embedding = await get_speaker_data(
            tts_manager.model,
            speaker_cache,
            voice_storage,
            audio_file=wav_file
        )
        
        # Сохраняем голос в хранилище
        voice_storage.save_voice(
            voice_id=voice_id,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            name=name,
            description=description
        )
        
        return {
            "status": "success",
            "message": f"Voice saved with ID: {voice_id}",
            "voice_id": voice_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to save voice: {str(e)}"
        )

@router.get("/cache/stats")
async def get_cache_stats(
    speaker_cache: SpeakerCache = Depends(get_speaker_cache)
):
    """Получение статистики работы кэша голосов"""
    return speaker_cache.get_stats()

@router.post("/cache/clear")
async def clear_cache(
    speaker_cache: SpeakerCache = Depends(get_speaker_cache)
):
    """Очистка кэша голосов"""
    speaker_cache.clear()
    return {"status": "success", "message": "Cache cleared"}

@router.get("/stream_status/{session_id}")
async def get_stream_status(session_id: str):
    """Получает статус сессии потоковой передачи"""
    if session_id in active_sessions:
        return active_sessions[session_id]
    else:
        return {"status": "not_found"}

@router.delete("/stream_cancel/{session_id}")
async def cancel_stream(session_id: str):
    """Отменяет текущую сессию потоковой передачи"""
    if session_id in active_sessions:
        active_sessions[session_id]["status"] = "canceled"
        return {"status": "canceled"}
    else:
        return {"status": "not_found"}

@router.get("/available_speakers")
async def get_available_speakers(
    tts_manager: TTSManager = Depends(get_tts_manager)
):
    """Получает доступные встроенные голоса из модели"""
    return tts_manager.get_available_speakers()

@router.get("/languages")
async def get_languages(
    tts_manager: TTSManager = Depends(get_tts_manager)
):
    """Получает поддерживаемые языки"""
    return tts_manager.get_available_languages()

@router.get("/health")
async def health_check(
    tts_manager: TTSManager = Depends(get_tts_manager)
):
    """Эндпоинт проверки работоспособности"""
    return {
        "status": "ok",
        "active_sessions": len(active_sessions),
        "model_loaded": tts_manager.model_loaded,
        "device": str(tts_manager.model.device if tts_manager.model_loaded else "unknown"),
        "max_concurrent_requests": tts_manager.semaphore._value if tts_manager.semaphore else 0
    }