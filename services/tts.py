import os
import torch
import asyncio
import traceback
from typing import Dict, Any, AsyncGenerator, Optional, Tuple, List
from fastapi import HTTPException

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager
from transformers import GenerationConfig

from config import device, CUSTOM_MODEL_PATH, configure_model_optimizations
from services.audio import (
    postprocess,
    generate_silence,
    encode_audio_common,
    preprocess_text,
    normalize_audio,
    resample_audio,
    mix_audio,
    apply_fade
)
from services.utils import active_sessions, _cleanup_session

# Оптимизированные параметры генерации для более быстрого вывода
OPTIMIZED_GENERATION_CONFIG = GenerationConfig(
    do_sample=False,
    num_beams=1,
    do_stream=True,
    temperature=1.0,  # Более низкая температура для более быстрых, детерминированных выводов
    repetition_penalty=1.0  # Стандартное значение, регулируйте при необходимости
)
# Добавляем XttsConfig в список безопасных глобальных объектов
torch.serialization.add_safe_globals([XttsConfig])

class TTSManager:
    """Менеджер для работы с TTS-моделью"""
    
    def __init__(self):
        """Инициализация менеджера TTS (без загрузки модели)"""
        self.model = None
        self.config = None
        self.semaphore = None
        self.model_loaded = False
    
    def initialize_model(self) -> Tuple[Xtts, XttsConfig]:
        """
        Инициализация TTS модели
        
        Returns:
            Кортеж (модель, конфигурация)
        """
        print("Loading XTTS model...", flush=True)
        
        # Загрузка конфигурации модели
        if os.path.exists(CUSTOM_MODEL_PATH) and os.path.isfile(CUSTOM_MODEL_PATH + "/config.json"):
            model_path = CUSTOM_MODEL_PATH
            print(f"Loading custom model from {model_path}", flush=True)
        else:
            print("Loading default model", flush=True)
            model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
            print(f"Downloading XTTS Model: {model_name}", flush=True)
            ModelManager().download_model(model_name)
            model_path = os.path.join(get_user_data_dir(
                "tts"), model_name.replace("/", "--"))
            print("XTTS Model downloaded", flush=True)

        print("Loading XTTS", flush=True)
        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config, 
            checkpoint_dir=model_path, 
            eval=True,
            use_deepspeed=True if device.type == "cuda" else False
        )
        model.to(device.type)
        
        # Предзагрузка фонемизаторов для распространенных языков
        if hasattr(model, "phonemizer") and hasattr(model.phonemizer, "preload_languages"):
            print("Preloading language phonemizers...", flush=True)
            common_languages = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]
            model.phonemizer.preload_languages(common_languages)
        
        print("XTTS Loaded.", flush=True)
        self.model = model
        self.config = config
        self.model_loaded = True
        return model, config
    
    def set_semaphore(self, max_concurrent_requests: int) -> None:
        """
        Устанавливает семафор для ограничения одновременных запросов
        
        Args:
            max_concurrent_requests: Максимальное количество одновременных запросов
        """
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    async def generate_stream(
        self, 
        session_id: str, 
        text: str, 
        language: str, 
        gpt_cond_latent: torch.Tensor, 
        speaker_embedding: torch.Tensor, 
        stream_chunk_size: int = 20, 
        add_wav_header: bool = True
    ) -> AsyncGenerator[bytes, None]:
        """
        Генерирует аудио-фрагменты потока для данной сессии
        
        Args:
            session_id: ID сессии
            text: Текст для синтеза
            language: Код языка
            gpt_cond_latent: Латентное представление голоса
            speaker_embedding: Вложение голоса
            stream_chunk_size: Размер фрагмента потока
            add_wav_header: Добавлять ли WAV-заголовок
            normalize: Нормализовать ли аудио
            target_sample_rate: Целевая частота дискретизации
            apply_fade_in_out: Применять ли эффект затухания в начале и конце
            
        Yields:
            Аудио-фрагменты в байтах
        """
        if not self.model_loaded:
            raise RuntimeError("TTS model is not loaded")
        
        try:
            # Сохраняем информацию о сессии
            active_sessions[session_id] = {
                "status": "processing",
                "text": text,
                "language": language,
                "created_at": asyncio.get_event_loop().time()
            }
            
            print(f"[{session_id}] Starting TTS stream generation with text: '{text[:30]}...'", flush=True)

            # Получаем семафор только для настройки вывода модели
            async with self.semaphore:
                # Проверяем и логируем размерность тензоров перед инференсом
                #print(f"[{session_id}] gpt_cond_latent dimensions: {gpt_cond_latent.dim()}, "f"shape: {list(gpt_cond_latent.shape)}", flush=True)
                
                # Отладочная печать атрибутов config (можно оставить для проверки)
                #print(f"[{session_id}] Global config attributes before inference: {vars(OPTIMIZED_GENERATION_CONFIG)}", flush=True)
                
                # Очищаем кэш модели перед генерацией (оставляем, если нужно)
                if hasattr(self.model, "clear_cache"):
                    self.model.clear_cache()
                    print(f"[{session_id}] Model cache cleared", flush=True)
                
                # Настраиваем генератор потока с глобальной конфигурацией
                chunks_generator = self.model.inference_stream(
                    text,
                    language,
                    gpt_cond_latent,
                    speaker_embedding,
                    stream_chunk_size=stream_chunk_size,
                    enable_text_splitting=True,
                    generation_config=OPTIMIZED_GENERATION_CONFIG # <-- Передаем глобальную
                )
                print(f"[{session_id}] Inference stream initialized", flush=True)
            
            # Обрабатываем фрагменты без блокировки семафора
            audio_chunk_count = 0
            for i, chunk in enumerate(chunks_generator):
                audio_chunk_count += 1
                
                # Применяем обработку аудио
                chunk = postprocess(chunk)
                
                if i == 0 and add_wav_header:
                    yield encode_audio_common(b"", encode_base64=False)
                    yield chunk.tobytes()
                else:
                    yield chunk.tobytes()
                
                # Проверяем, была ли сессия отменена
                if session_id in active_sessions and active_sessions[session_id]["status"] == "canceled":
                    print(f"[{session_id}] Stream generation canceled", flush=True)
                    return
                
                # Периодически логируем прогресс
                if audio_chunk_count % 10 == 0:
                    print(f"[{session_id}] Generated {audio_chunk_count} audio chunks", flush=True)

            # Обновляем статус сессии по завершении
            if session_id in active_sessions:
                active_sessions[session_id]["status"] = "completed"
            
            print(f"[{session_id}] Stream generation completed, generated {audio_chunk_count} audio chunks", flush=True)
        
        except Exception as e:
            # Обновляем статус сессии при ошибке
            if session_id in active_sessions:
                active_sessions[session_id]["status"] = "error"
                active_sessions[session_id]["error"] = str(e)
            
            # Подробное логирование ошибки
            print(f"[{session_id}] Error in stream_generator: {str(e)}", flush=True)
            print(f"[{session_id}] Error traceback: {traceback.format_exc()}", flush=True)
            
            # Если ошибка связана с размерностью тензоров, логируем их формы
            if "dimensions" in str(e) and "gpt_cond_latent" in locals():
                print(f"[{session_id}] Error with tensor dimensions. "
                     f"gpt_cond_latent shape: {list(gpt_cond_latent.shape)}, "
                     f"speaker_embedding shape: {list(speaker_embedding.shape)}", flush=True)
            
            # Очищаем кэш модели при ошибке
            if hasattr(self.model, "clear_cache"):
                self.model.clear_cache()
                print(f"[{session_id}] Model cache cleared after error", flush=True)
            
            raise HTTPException(status_code=500, detail=str(e))
        
        finally:
            # Очищаем сессию
            asyncio.create_task(_cleanup_session(session_id))
    
    async def synthesize_full(
        self, 
        text: str, 
        language: str, 
        gpt_cond_latent: torch.Tensor, 
        speaker_embedding: torch.Tensor
    ) -> bytes:
        """
        Синтезирует полное аудио (не в потоковом режиме)
        
        Args:
            text: Текст для синтеза
            language: Код языка
            gpt_cond_latent: Латентное представление голоса
            speaker_embedding: Вложение голоса
            
        Returns:
            Аудиоданные в формате WAV (байты)
        """
        if not self.model_loaded:
            raise RuntimeError("TTS model is not loaded")
        
        try:
            async with self.semaphore:
                # Генерируем речь
                out = self.model.inference(
                    text,
                    language,
                    gpt_cond_latent,
                    speaker_embedding,
                    generation_config=OPTIMIZED_GENERATION_CONFIG
                )

                wav = postprocess(torch.tensor(out["wav"]))
                return encode_audio_common(wav.tobytes())
        except Exception as e:
            print(f"Error in synthesize_full: {str(e)}", flush=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error generating speech: {str(e)}"
            )
    
    def get_available_languages(self) -> List[str]:
        """
        Получает список поддерживаемых языков
        
        Returns:
            Список кодов языков
        """
        if not self.model_loaded:
            raise RuntimeError("TTS model is not loaded")
        
        return self.config.languages
    
    def get_available_speakers(self) -> Dict[str, Dict[str, Any]]:
        """
        Получает список доступных встроенных голосов
        
        Returns:
            Словарь с информацией о голосах
        """
        if not self.model_loaded:
            raise RuntimeError("TTS model is not loaded")
        
        if hasattr(self.model, "speaker_manager") and hasattr(self.model.speaker_manager, "speakers"):
            return {
                speaker_id: {
                    "name": speaker_id,
                }
                for speaker_id in self.model.speaker_manager.speakers.keys()
            }
        else:
            return {}