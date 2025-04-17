import io
import wave
import base64
import numpy as np
import torch
from typing import Union, Tuple, List, Optional
from config import (
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    CHANNELS,
    MAX_TEXT_CHUNK_SIZE,
    MIN_TEXT_CHUNK_SIZE
)

def postprocess(wav: Union[torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
    """
    Post-process output waveform
    
    Args:
        wav: Tensor or list of tensors with audio data
        
    Returns:
        Processed audio data as numpy array
    """
    if isinstance(wav, list):
        wav = torch.cat(wav, dim=0)
    wav = wav.clone().detach().cpu().numpy()
    wav = wav[None, : int(wav.shape[0])]
    wav = np.clip(wav, -1, 1)
    wav = (wav * 32767).astype(np.int16)
    return wav

def encode_audio_common(
    frame_input: bytes, 
    encode_base64: bool = True, 
    sample_rate: int = SAMPLE_RATE, 
    sample_width: int = SAMPLE_WIDTH, 
    channels: int = CHANNELS
) -> Union[str, bytes]:
    """
    Encodes audio data to WAV format and optionally to base64
    
    Args:
        frame_input: Input audio data in bytes
        encode_base64: Flag for encoding result to base64
        sample_rate: Sampling rate
        sample_width: Sample depth in bytes
        channels: Number of channels
        
    Returns:
        Encoded audio data (base64 string or bytes)
    """
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    if encode_base64:
        b64_encoded = base64.b64encode(wav_buf.getbuffer()).decode("utf-8")
        return b64_encoded
    else:
        return wav_buf.read()

def generate_silence(duration_ms: int = 100, sample_rate: int = SAMPLE_RATE) -> bytes:
    """
    Generates audio silence of specified duration
    
    Args:
        duration_ms: Duration in milliseconds
        sample_rate: Sampling rate
        
    Returns:
        Silence audio data in bytes
    """
    num_samples = int(duration_ms * sample_rate / 1000)
    silence = np.zeros(num_samples, dtype=np.int16)
    return silence.tobytes()

def preprocess_text(text: str, max_chunk_length: int = MAX_TEXT_CHUNK_SIZE) -> List[str]:
    """
    Preprocess text for more efficient streaming
    
    Splits long paragraphs into smaller, manageable chunks that
    can be more efficiently processed by the TTS model.
    
    Args:
        text: Source text
        max_chunk_length: Maximum text chunk length
        
    Returns:
        List of text chunks
    """
    # Split text by sentence boundaries
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Group sentences into reasonably sized chunks
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would make the chunk too long, start a new chunk
        if len(current_chunk) + len(sentence) > max_chunk_length:
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

def ensure_tensor_dimensions(tensor: Union[torch.Tensor, List]) -> torch.Tensor:
    """
    Checks and fixes tensor dimensions for compatibility with XTTS model
    
    GPT-conditioning latent tensor should have dimensions [1, n, d],
    where n is the number of vectors, d is the size of each vector.
    
    Args:
        tensor: Input tensor or list
        
    Returns:
        Tensor with correct dimensions
    """
    from config import device
    
    # Check if input is a tensor
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor).to(device)
    
    # Get current tensor dimensions
    dims = tensor.dim()
    
    # Fix dimensions if necessary
    if dims == 1:
        # [d] -> [1, 1, d]
        return tensor.unsqueeze(0).unsqueeze(0)
    elif dims == 2:
        # [n, d] -> [1, n, d]
        return tensor.unsqueeze(0)
    elif dims == 3:
        # [1, n, d] - correct dimension
        return tensor
    else:
        # Unexpected dimension
        print(f"Warning: Unexpected tensor dimensions: {tensor.size()}", flush=True)
        # Try to fix if possible
        if dims > 3:
            return tensor.squeeze(0) if tensor.size(0) == 1 else tensor[:1]
        return tensor

def ensure_speaker_embedding_dimensions(speaker_embedding: torch.Tensor) -> torch.Tensor:
    """
    Fixes speaker_embedding dimensions for use in the model
    
    Args:
        speaker_embedding: Voice embedding tensor
        
    Returns:
        Tensor with correct dimensions [1, 512]
    """
    if speaker_embedding.dim() == 1:
        return speaker_embedding.unsqueeze(0)  # [512] -> [1, 512]
    return speaker_embedding

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalizes audio data to range [-1, 1]
    
    Args:
        audio: Audio data as numpy array
        
    Returns:
        Normalized audio data
    """
    return audio / np.max(np.abs(audio))

def resample_audio(
    audio: np.ndarray,
    original_rate: int,
    target_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Resamples audio data to target sampling rate
    
    Args:
        audio: Audio data as numpy array
        original_rate: Original sampling rate
        target_rate: Target sampling rate
        
    Returns:
        Resampled audio data
    """
    if original_rate == target_rate:
        return audio
    
    # Calculate resampling ratio
    ratio = target_rate / original_rate
    
    # Calculate new length
    new_length = int(len(audio) * ratio)
    
    # Create time scale for interpolation
    old_indices = np.arange(len(audio))
    new_indices = np.linspace(0, len(audio) - 1, new_length)
    
    # Linear interpolation
    return np.interp(new_indices, old_indices, audio)

def mix_audio(audio1: np.ndarray, audio2: np.ndarray, ratio: float = 0.5) -> np.ndarray:
    """
    Mixes two audio signals with given ratio
    
    Args:
        audio1: First audio signal
        audio2: Second audio signal
        ratio: Mixing ratio (0.0 - only audio1, 1.0 - only audio2)
        
    Returns:
        Mixed audio signal
    """
    # Normalize signals
    audio1 = normalize_audio(audio1)
    audio2 = normalize_audio(audio2)
    
    # Mix with given ratio
    mixed = audio1 * (1 - ratio) + audio2 * ratio
    
    # Normalize result
    return normalize_audio(mixed)

def apply_fade(audio: np.ndarray, fade_length: int = 100) -> np.ndarray:
    """
    Applies smooth fade to the beginning and end of audio signal
    
    Args:
        audio: Audio signal
        fade_length: Fade length in samples
        
    Returns:
        Audio signal with applied fade
    """
    # Create fade mask
    fade_in = np.linspace(0, 1, fade_length)
    fade_out = np.linspace(1, 0, fade_length)
    
    # Apply fade
    audio[:fade_length] *= fade_in
    audio[-fade_length:] *= fade_out
    
    return audio