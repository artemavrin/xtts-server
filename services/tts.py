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

# Optimized generation parameters for faster output
OPTIMIZED_GENERATION_CONFIG = GenerationConfig(
    do_sample=False,
    num_beams=1,
    do_stream=True,
    temperature=1.0,  # Lower temperature for faster, more deterministic outputs
    repetition_penalty=1.0  # Standard value, adjust as needed
)
# Add XttsConfig to the list of safe global objects
torch.serialization.add_safe_globals([XttsConfig])

class TTSManager:
    """Manager for working with TTS model"""
    
    def __init__(self):
        """Initialize TTS manager (without loading the model)"""
        self.model = None
        self.config = None
        self.semaphore = None
        self.model_loaded = False
    
    def initialize_model(self) -> Tuple[Xtts, XttsConfig]:
        """
        Initialize TTS model
        
        Returns:
            Tuple (model, configuration)
        """
        print("Loading XTTS model...", flush=True)
        
        # Load model configuration
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
        
        # Preload phonemizers for common languages
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
        Set semaphore to limit concurrent requests
        
        Args:
            max_concurrent_requests: Maximum number of concurrent requests
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
        Generate audio stream chunks for the given session
        
        Args:
            session_id: Session ID
            text: Text to synthesize
            language: Language code
            gpt_cond_latent: Voice latent representation
            speaker_embedding: Voice embedding
            stream_chunk_size: Stream chunk size
            add_wav_header: Whether to add WAV header
            normalize: Whether to normalize audio
            target_sample_rate: Target sampling rate
            apply_fade_in_out: Whether to apply fade effect at start and end
            
        Yields:
            Audio chunks in bytes
        """
        if not self.model_loaded:
            raise RuntimeError("TTS model is not loaded")
        
        try:
            # Save session information
            active_sessions[session_id] = {
                "status": "processing",
                "text": text,
                "language": language,
                "created_at": asyncio.get_event_loop().time()
            }
            
            print(f"[{session_id}] Starting TTS stream generation with text: '{text[:30]}...'", flush=True)

            # Get semaphore only for model output setup
            async with self.semaphore:
                # Check and log tensor dimensions before inference
                #print(f"[{session_id}] gpt_cond_latent dimensions: {gpt_cond_latent.dim()}, "f"shape: {list(gpt_cond_latent.shape)}", flush=True)
                
                # Debug print of config attributes (can be left for verification)
                #print(f"[{session_id}] Global config attributes before inference: {vars(OPTIMIZED_GENERATION_CONFIG)}", flush=True)
                
                # Clear model cache before generation (leave if needed)
                if hasattr(self.model, "clear_cache"):
                    self.model.clear_cache()
                    print(f"[{session_id}] Model cache cleared", flush=True)
                
                # Configure stream generator with global configuration
                chunks_generator = self.model.inference_stream(
                    text,
                    language,
                    gpt_cond_latent,
                    speaker_embedding,
                    stream_chunk_size=stream_chunk_size,
                    enable_text_splitting=True,
                    generation_config=OPTIMIZED_GENERATION_CONFIG # <-- Pass global
                )
                print(f"[{session_id}] Inference stream initialized", flush=True)
            
            # Process chunks without semaphore blocking
            audio_chunk_count = 0
            for i, chunk in enumerate(chunks_generator):
                audio_chunk_count += 1
                
                # Apply audio processing
                chunk = postprocess(chunk)
                
                if i == 0 and add_wav_header:
                    yield encode_audio_common(b"", encode_base64=False)
                    yield chunk.tobytes()
                else:
                    yield chunk.tobytes()
                
                # Check if session was canceled
                if session_id in active_sessions and active_sessions[session_id]["status"] == "canceled":
                    print(f"[{session_id}] Stream generation canceled", flush=True)
                    return
                
                # Periodically log progress
                if audio_chunk_count % 10 == 0:
                    print(f"[{session_id}] Generated {audio_chunk_count} audio chunks", flush=True)

            # Update session status on completion
            if session_id in active_sessions:
                active_sessions[session_id]["status"] = "completed"
            
            print(f"[{session_id}] Stream generation completed, generated {audio_chunk_count} audio chunks", flush=True)
        
        except Exception as e:
            # Update session status on error
            if session_id in active_sessions:
                active_sessions[session_id]["status"] = "error"
                active_sessions[session_id]["error"] = str(e)
            
            # Detailed error logging
            print(f"[{session_id}] Error in stream_generator: {str(e)}", flush=True)
            print(f"[{session_id}] Error traceback: {traceback.format_exc()}", flush=True)
            
            # If error is related to tensor dimensions, log their shapes
            if "dimensions" in str(e) and "gpt_cond_latent" in locals():
                print(f"[{session_id}] Error with tensor dimensions. "
                     f"gpt_cond_latent shape: {list(gpt_cond_latent.shape)}, "
                     f"speaker_embedding shape: {list(speaker_embedding.shape)}", flush=True)
            
            # Clear model cache on error
            if hasattr(self.model, "clear_cache"):
                self.model.clear_cache()
                print(f"[{session_id}] Model cache cleared after error", flush=True)
            
            raise HTTPException(status_code=500, detail=str(e))
        
        finally:
            # Clean up session
            asyncio.create_task(_cleanup_session(session_id))
    
    async def synthesize_full(
        self, 
        text: str, 
        language: str, 
        gpt_cond_latent: torch.Tensor, 
        speaker_embedding: torch.Tensor
    ) -> bytes:
        """
        Synthesize full audio (non-streaming)
        
        Args:
            text: Text to synthesize
            language: Language code
            gpt_cond_latent: Voice latent representation
            speaker_embedding: Voice embedding
            
        Returns:
            Audio data in WAV format (bytes)
        """
        if not self.model_loaded:
            raise RuntimeError("TTS model is not loaded")
        
        try:
            async with self.semaphore:
                # Generate speech
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
        Get list of supported languages
        
        Returns:
            List of language codes
        """
        if not self.model_loaded:
            raise RuntimeError("TTS model is not loaded")
        
        return self.config.languages
    
    def get_available_speakers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get list of available built-in voices
        
        Returns:
            Dictionary with voice information
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