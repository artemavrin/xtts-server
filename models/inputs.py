from typing import List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import numpy as np

class TTSInputs(BaseModel):
    """
    Input parameters for text-to-speech generation
    
    Supports multiple ways to specify voice:
    1. By saved voice ID (speaker_id)
    2. By built-in model voice ID (speaker_id)
    3. By audio file uploaded via API (audio_file_path)
    4. By direct voice parameters (gpt_cond_latent, speaker_embedding)
    """
    text: str = Field(..., description="Text to synthesize")
    language: str = Field(..., description="Language code (e.g., 'en', 'ru')")
    speaker_id: Optional[str] = Field(None, description="Voice ID (saved or built-in)")
    audio_file_path: Optional[str] = Field(None, description="Path to audio file for voice cloning")
    gpt_cond_latent: Optional[List[List[float]]] = Field(None, description="GPT conditioning latent")
    speaker_embedding: Optional[List[float]] = Field(None, description="Speaker embedding vector")

    @field_validator('text')
    def validate_text(cls, v):
        """Validate text length and content"""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        if len(v) > 10000:  # Arbitrary limit
            raise ValueError("Text is too long")
        return v

    @field_validator('language')
    def validate_language(cls, v):
        """Validate language code format"""
        if not v or len(v) != 2:
            raise ValueError("Language must be a 2-letter code")
        return v.lower()

class StreamingOptions(BaseModel):
    """Options for streaming audio generation"""
    stream_chunk_size: int = Field(100, description="Size of audio chunks in milliseconds")
    add_wav_header: bool = Field(True, description="Add WAV header to audio stream")

    @field_validator('stream_chunk_size')
    def validate_chunk_size(cls, v):
        """Validate chunk size"""
        if v < 10 or v > 1000:
            raise ValueError("Chunk size must be between 10 and 1000 ms")
        return v

class CloneResponseFormat(str, Enum):
    """Response format for voice cloning"""
    json = "json"           # Returns data in JSON
    save = "save"           # Saves voice to storage
    json_and_save = "both"  # Both returns JSON and saves

class CloneVoiceOptions(BaseModel):
    """Options for voice cloning"""
    response_format: str = Field("json", description="Response format: json, save, or both")
    save_id: Optional[str] = Field(None, description="ID to save voice under")
    name: Optional[str] = Field(None, description="Voice name")
    description: Optional[str] = Field(None, description="Voice description")

    @field_validator('response_format')
    def validate_format(cls, v):
        """Validate response format"""
        if v not in ["json", "save", "both"]:
            raise ValueError("Response format must be one of: json, save, both")
        return v

class BatchTTSInputs(BaseModel):
    """Input parameters for batch text-to-speech processing"""
    texts: List[str] = Field(..., description="List of texts to synthesize")
    language: str = Field(..., description="Language code")
    speaker_id: str = Field(..., description="Voice ID")
    options: Optional[StreamingOptions] = Field(None, description="Streaming options")

    @field_validator('texts')
    def validate_texts(cls, v):
        """Validate texts list"""
        if not v:
            raise ValueError("Texts list cannot be empty")
        if len(v) > 100:  # Arbitrary limit
            raise ValueError("Too many texts in batch")
        return v

class VoiceParameters(BaseModel):
    """Voice parameters for direct use in TTS"""
    gpt_cond_latent: List[List[float]] = Field(..., description="GPT conditioning latent")
    speaker_embedding: List[float] = Field(..., description="Speaker embedding vector")

    @field_validator('gpt_cond_latent')
    def validate_gpt_cond_latent(cls, v):
        """Validate GPT conditioning latent dimensions"""
        if not isinstance(v, list) or not all(isinstance(row, list) for row in v):
            raise ValueError("gpt_cond_latent must be a 2D list")
        return v

    @field_validator('speaker_embedding')
    def validate_speaker_embedding(cls, v):
        """Validate speaker embedding dimensions"""
        if not isinstance(v, list):
            raise ValueError("speaker_embedding must be a list")
        return v