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

# Global dictionary for tracking active sessions
active_sessions: Dict[str, Dict[str, Any]] = {}

def preprocess_text(text: str) -> List[str]:
    """
    Preprocess text for more efficient streaming
    
    Splits long paragraphs into smaller, manageable chunks that
    can be more efficiently processed by the TTS model.
    
    Args:
        text: Original text
        
    Returns:
        List of text chunks
    """
    # Split text by sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Group sentences into reasonable-sized chunks
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would make the chunk too long, start a new chunk
        if len(current_chunk) + len(sentence) > MAX_TEXT_CHUNK_SIZE:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def generate_silence(duration_ms: int = 100, sample_rate: int = SAMPLE_RATE) -> bytes:
    """
    Generates silence data for faster initial response
    
    Args:
        duration_ms: Silence duration in milliseconds
        sample_rate: Sampling rate
        
    Returns:
        Bytes of silence audio data
    """
    num_samples = int(duration_ms * sample_rate / 1000)
    silence = np.zeros(num_samples, dtype=np.int16)
    return silence.tobytes()

async def _cleanup_session(sid: str, delay: int = 60) -> None:
    """
    Clean up session after delay
    
    Args:
        sid: Session ID
        delay: Delay before cleanup in seconds
    """
    await asyncio.sleep(delay)
    if sid in active_sessions:
        del active_sessions[sid]
    print(f"Cleaned up session {sid}", flush=True)

def generate_session_id() -> str:
    """
    Generates a unique session ID
    
    Returns:
        Unique session ID
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
    Get voice data from different sources with caching optimization
    
    Args:
        model: TTS model
        speaker_cache: Voice cache
        voice_storage: Voice storage (optional)
        speaker_id: Voice ID
        audio_file: Uploaded audio file
        audio_file_path: Path to audio file on server
        
    Returns:
        Tuple (gpt_cond_latent, speaker_embedding)
        
    Raises:
        HTTPException: If voice not found or error occurred during processing
    """
    try:
        # Check cache
        if speaker_id:
            cached_data = speaker_cache.get(speaker_id)
            if cached_data is not None:
                print(f"Using cached voice data for speaker_id: {speaker_id}", flush=True)
                return cached_data
        
        # Get voice data
        if speaker_id:
            if voice_storage and voice_storage.has_voice(speaker_id):
                print(f"Loading voice data from storage for speaker_id: {speaker_id}", flush=True)
                voice_data = voice_storage.get_voice(speaker_id)
                if voice_data is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Voice data not found for speaker_id: {speaker_id}"
                    )
                # Convert data to tensors with correct dtype
                gpt_cond_latent = torch.tensor(voice_data["gpt_cond_latent"], dtype=torch.float).to(model.device)
                speaker_embedding = torch.tensor(voice_data["speaker_embedding"], dtype=torch.float).to(model.device)
                
                # Restore gpt_cond_latent tensor shape
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
        
        # Fix speaker_embedding dimensions (ensures minimum 2D: [1, 512])
        speaker_embedding = ensure_speaker_embedding_dimensions(speaker_embedding)
        
        # Add last dimension ONLY if it's missing (for Conv1d)
        if speaker_embedding.dim() == 2:
            speaker_embedding = speaker_embedding.unsqueeze(-1) # [1, 512] -> [1, 512, 1]
        # If already 3D ([1, 512, 1]), do nothing
        
        # Cache result (with shape [1, 512, 1])
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
    Converts synchronous generator to asynchronous generator
    
    Args:
        sync_generator: Synchronous generator or iterator
        
    Yields:
        Items from synchronous generator
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
    Preloads frequently used voices into cache for quick access
    
    Args:
        model: TTS model
        speaker_cache: Voice cache
        voice_storage: Voice storage
        voices_to_preload: List of voices to preload
        max_preload: Maximum number of voices to preload
    """
    print("Preloading voices to cache...", flush=True)
    
    # List of voices to preload
    if voices_to_preload is None:
        voices_to_preload = []
    
    # Add all voices from model if they exist
    if hasattr(model, "speaker_manager") and hasattr(model.speaker_manager, "speakers"):
        voices_to_preload.extend(list(model.speaker_manager.speakers.keys()))
    
    # Add saved voices if storage is initialized
    if voice_storage is not None:
        saved_voices = voice_storage.list_voices()
        voices_to_preload.extend(list(saved_voices.keys()))
    
    # Remove duplicates
    voices_to_preload = list(set(voices_to_preload))
    
    # Limit number of preloaded voices
    voices_to_preload = voices_to_preload[:max_preload]
    
    if not voices_to_preload:
        print("No voices to preload.", flush=True)
        return
    
    # Load voices into cache
    loaded_count = 0
    for voice_id in voices_to_preload:
        try:
            # Try to load from different sources
            
            # 1. Check saved voices
            if voice_storage is not None:
                saved_voice = voice_storage.get_voice(voice_id)
                if saved_voice:
                    # Convert data from JSON to tensors
                    gpt_cond_latent = torch.tensor(saved_voice["gpt_cond_latent"]).to(device)
                    speaker_embedding = torch.tensor(saved_voice["speaker_embedding"]).to(device)
                    
                    # If there's information about tensor shape, use it
                    if "gpt_cond_latent_shape" in saved_voice:
                        tensor_shape = saved_voice["gpt_cond_latent_shape"]
                        gpt_cond_latent = gpt_cond_latent.reshape(*tensor_shape)
                    
                    # Check and fix gpt_cond_latent dimensions
                    gpt_cond_latent = ensure_tensor_dimensions(gpt_cond_latent)
                    
                    # Add to cache
                    speaker_cache.set(f"speaker_id:{voice_id}", (gpt_cond_latent, speaker_embedding))
                    loaded_count += 1
                    continue
            
            # 2. Check built-in voices of the model
            if hasattr(model, "speaker_manager") and voice_id in model.speaker_manager.speakers:
                speaker_data = model.speaker_manager.speakers[voice_id]
                gpt_cond_latent = speaker_data["gpt_cond_latent"]
                speaker_embedding = speaker_data["speaker_embedding"]
                
                # Check and fix gpt_cond_latent dimensions
                gpt_cond_latent = ensure_tensor_dimensions(gpt_cond_latent)
                
                # Add to cache
                speaker_cache.set(f"speaker_id:{voice_id}", (gpt_cond_latent, speaker_embedding))
                loaded_count += 1
        
        except Exception as e:
            print(f"Error preloading voice {voice_id}: {str(e)}", flush=True)
    
    print(f"Preloaded {loaded_count}/{len(voices_to_preload)} voices to cache.", flush=True)

async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Saves uploaded file to temporary directory
    
    Args:
        upload_file: Uploaded file
        
    Returns:
        Path to saved file
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = temp_file.name
            
            # Read and write file content
            content = await upload_file.read()
            temp_file.write(content)
            
            return temp_path
    except Exception as e:
        print(f"Error saving upload file: {str(e)}", flush=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error saving upload file: {str(e)}"
        )