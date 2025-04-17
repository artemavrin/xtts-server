import torch
import uuid
import asyncio
from fastapi import APIRouter, UploadFile, HTTPException, Depends
from fastapi.responses import StreamingResponse

from config import (
    OPTIMIZED_GENERATION_CONFIG,
    STREAM_CHUNK_SIZE,
    ADD_WAV_HEADER,
    MAX_TEXT_CHUNK_SIZE,
    MIN_TEXT_CHUNK_SIZE,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    CHANNELS
)
from models.inputs import TTSInputs, StreamingOptions, CloneVoiceOptions
from models.voice_storage import VoiceStorage
from services.cache import SpeakerCache
from services.tts import TTSManager
from services.utils import (
    get_speaker_data,
    generate_session_id,
    active_sessions,
    preprocess_text,
    generate_silence
)
from services.audio import (
    ensure_tensor_dimensions,
    ensure_speaker_embedding_dimensions,
    encode_audio_common
)

router = APIRouter()

# Dependencies for service injection
def get_tts_manager():
    """Get TTS manager"""
    from main import tts_manager
    return tts_manager

def get_voice_storage():
    """Get voice storage"""
    from main import voice_storage
    return voice_storage

def get_speaker_cache():
    """Get voice cache"""
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
    Stream text to speech
    
    Supports multiple ways to specify voice:
    1. By saved voice ID (speaker_id)
    2. By built-in model voice ID (speaker_id)
    3. By audio file uploaded via API (audio_file_path)
    4. By direct voice parameters (gpt_cond_latent, speaker_embedding)
    """
    # Generate session ID
    session_id = generate_session_id()
    
    # Initialize session
    active_sessions[session_id] = {
        "status": "initializing", 
        "text": request.text,
        "language": request.language,
        "speaker_id": request.speaker_id,
        "created_at": 0  # Will be updated when generation starts
    }
    
    print(f"[{session_id}] TTS Stream requested for text: '{request.text[:50]}...' with language: {request.language}", flush=True)
    
    async def stream_generator():
        try:
            # Check if voice parameters were provided directly
            if request.gpt_cond_latent is not None and request.speaker_embedding is not None:
                print(f"[{session_id}] Using provided voice embeddings", flush=True)
                # Convert lists to tensors
                gpt_cond_latent = torch.tensor(request.gpt_cond_latent).to(tts_manager.model.device)
                speaker_embedding = torch.tensor(request.speaker_embedding).to(tts_manager.model.device)
                
                # Check and fix dimensions
                gpt_cond_latent = ensure_tensor_dimensions(gpt_cond_latent)
                speaker_embedding = ensure_speaker_embedding_dimensions(speaker_embedding)
                new_shape = gpt_cond_latent.shape
                
                print(f"[{session_id}] Voice tensor shape: {gpt_cond_latent.shape} -> {new_shape}", flush=True)
            else:
                # Get speaker data from other sources
                print(f"[{session_id}] Fetching voice data for speaker_id: {request.speaker_id}", flush=True)
                gpt_cond_latent, speaker_embedding = await get_speaker_data(
                    tts_manager.model,
                    speaker_cache,
                    voice_storage,
                    speaker_id=request.speaker_id,
                    audio_file_path=request.audio_file_path
                )
                
                # Log tensor shapes
                print(f"[{session_id}] Voice data fetched. gpt_cond_latent shape: {gpt_cond_latent.shape}, "
                     f"speaker_embedding shape: {speaker_embedding.shape}", flush=True)
            
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
            # Handle errors
            if session_id in active_sessions:
                active_sessions[session_id]["status"] = "error"
                active_sessions[session_id]["error"] = str(e)
            
            # Detailed error logging
            print(f"[{session_id}] Error in stream_generator: {str(e)}", flush=True)
            import traceback
            print(f"[{session_id}] Error traceback: {traceback.format_exc()}", flush=True)
            
            raise
    
    # Return streaming response with async generator
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
    Generate full audio (non-streaming)
    
    Supports multiple ways to specify voice:
    1. By saved voice ID (speaker_id)
    2. By built-in model voice ID (speaker_id)
    3. By audio file uploaded via API (audio_file_path)
    4. By direct voice parameters (gpt_cond_latent, speaker_embedding)
    """
    try:
        # Check if voice parameters were provided directly
        if request.gpt_cond_latent is not None and request.speaker_embedding is not None:
            # Convert lists to tensors
            gpt_cond_latent = torch.tensor(request.gpt_cond_latent).to(tts_manager.model.device)
            speaker_embedding = torch.tensor(request.speaker_embedding).to(tts_manager.model.device)
            
            # Check and fix dimensions
            gpt_cond_latent = ensure_tensor_dimensions(gpt_cond_latent)
        else:
            # Get speaker data from other sources
            gpt_cond_latent, speaker_embedding = await get_speaker_data(
                tts_manager.model,
                speaker_cache,
                voice_storage,
                speaker_id=request.speaker_id,
                audio_file_path=request.audio_file_path
            )

        # Generate speech
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
    Clone voice from audio file
    
    Supports various response formats:
    - json: returns voice data in JSON
    - save: saves voice to storage and returns ID
    - both: both saves voice and returns data in JSON
    """
    try:
        # If options not specified, use default values
        if options is None:
            options = CloneVoiceOptions()
        
        # Check if file is audio
        if not wav_file.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=400, 
                detail=f"Uploaded file is not an audio file. Content type: {wav_file.content_type}"
            )

        # Get voice data from audio file
        gpt_cond_latent, speaker_embedding = await get_speaker_data(
            tts_manager.model,
            speaker_cache,
            voice_storage,
            audio_file=wav_file
        )
        
        # Prepare voice data for JSON response
        voice_data = {
            "gpt_cond_latent": gpt_cond_latent.cpu().squeeze().tolist(),
            "speaker_embedding": speaker_embedding.cpu().squeeze(-1).tolist(),
        }
        
        # Process different response formats
        if options.response_format == "json":
            # Just return voice data
            return voice_data
            
        elif options.response_format == "save":
            # Save voice to storage
            if not options.save_id:
                # If ID not specified, generate random
                options.save_id = f"voice-{uuid.uuid4().hex[:8]}"
                
            # Save voice
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
            # Both save and return data
            if not options.save_id:
                # If ID not specified, generate random
                options.save_id = f"voice-{uuid.uuid4().hex[:8]}"
                
            # Save voice
            voice_storage.save_voice(
                options.save_id,
                gpt_cond_latent,
                speaker_embedding,
                options.name or options.save_id,
                options.description
            )
            
            # Return both voice data and save information
            return {
                "status": "success",
                "message": f"Voice cloned and saved successfully",
                "voice_id": options.save_id,
                "voice_data": voice_data
            }
    
    except Exception as e:
        # More informative error handling
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
    """Returns list of all saved voices"""
    return voice_storage.list_voices()

@router.delete("/saved_voices/{voice_id}")
async def delete_saved_voice(
    voice_id: str,
    voice_storage: VoiceStorage = Depends(get_voice_storage)
):
    """Delete saved voice"""
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
    """Clone voice from audio file and save it for later use"""
    try:
        # Get voice data from audio file
        gpt_cond_latent, speaker_embedding = await get_speaker_data(
            tts_manager.model,
            speaker_cache,
            voice_storage,
            audio_file=wav_file
        )
        
        # Save voice to storage
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
    """Get voice cache statistics"""
    return speaker_cache.get_stats()

@router.post("/cache/clear")
async def clear_cache(
    speaker_cache: SpeakerCache = Depends(get_speaker_cache)
):
    """Clear voice cache"""
    speaker_cache.clear()
    return {"status": "success", "message": "Cache cleared"}

@router.get("/stream_status/{session_id}")
async def get_stream_status(session_id: str):
    """Get streaming session status"""
    if session_id in active_sessions:
        return active_sessions[session_id]
    else:
        return {"status": "not_found"}

@router.delete("/stream_cancel/{session_id}")
async def cancel_stream(session_id: str):
    """Cancel current streaming session"""
    if session_id in active_sessions:
        active_sessions[session_id]["status"] = "canceled"
        return {"status": "canceled"}
    else:
        return {"status": "not_found"}

@router.get("/available_speakers")
async def get_available_speakers(
    tts_manager: TTSManager = Depends(get_tts_manager)
):
    """Get available built-in voices from model"""
    return tts_manager.get_available_speakers()

@router.get("/languages")
async def get_languages(
    tts_manager: TTSManager = Depends(get_tts_manager)
):
    """Get supported languages"""
    return tts_manager.get_available_languages()

@router.get("/health")
async def health_check(
    tts_manager: TTSManager = Depends(get_tts_manager)
):
    """Health check endpoint"""
    return {
        "status": "ok",
        "active_sessions": len(active_sessions),
        "model_loaded": tts_manager.model_loaded,
        "device": str(tts_manager.model.device if tts_manager.model_loaded else "unknown"),
        "max_concurrent_requests": tts_manager.semaphore._value if tts_manager.semaphore else 0
    }

@router.post("/tts_stream_optimized")
async def tts_stream_optimized(
    request: TTSInputs, 
    options: StreamingOptions = StreamingOptions(),
    tts_manager: TTSManager = Depends(get_tts_manager),
    voice_storage: VoiceStorage = Depends(get_voice_storage),
    speaker_cache: SpeakerCache = Depends(get_speaker_cache)
):
    """
    Optimized text to speech streaming
    
    Main optimizations:
    - Text preprocessing
    - Optimized generation parameters
    - Improved error handling
    - Generation cancellation support
    """
    session_id = generate_session_id()
    
    # Initialize session
    active_sessions[session_id] = {
        "status": "initializing",
        "text": request.text,
        "language": request.language,
        "created_at": asyncio.get_event_loop().time()
    }
    
    async def optimized_stream():
        try:
            # Send WAV header immediately
            if options.add_wav_header:
                yield encode_audio_common(b"", encode_base64=False)
            
            # Send small amount of silence to establish connection
            yield generate_silence(50)  # 50ms of silence
            
            # Process speaker data in background
            gpt_cond_latent, speaker_embedding = await get_speaker_data(
                tts_manager.model,
                speaker_cache,
                voice_storage,
                speaker_id=request.speaker_id,
                audio_file_path=request.audio_file_path
            )
            
            # Update session status
            if session_id in active_sessions:
                active_sessions[session_id]["status"] = "processing"
            
            # Preprocess text for faster generation
            text_chunks = preprocess_text(request.text)
            
            # Process chunks with minimal delay between them
            for chunk_idx, chunk_text in enumerate(text_chunks):
                # Get semaphore only for generation setup
                async with tts_manager.semaphore:
                    chunks_generator = tts_manager.model.inference_stream(
                        chunk_text,
                        request.language,
                        gpt_cond_latent,
                        speaker_embedding,
                        stream_chunk_size=options.stream_chunk_size,
                        enable_text_splitting=True,
                        generation_config=OPTIMIZED_GENERATION_CONFIG
                    )
                
                # Process generated audio chunks
                for audio_chunk in chunks_generator:
                    # Check for cancellation
                    if session_id in active_sessions and active_sessions[session_id]["status"] == "canceled":
                        return
                    
                    # Process and yield audio
                    processed_chunk = tts_manager.postprocess(audio_chunk)
                    yield processed_chunk.tobytes()
            
            # Mark as completed
            if session_id in active_sessions:
                active_sessions[session_id]["status"] = "completed"
                
        except Exception as e:
            # Handle errors
            if session_id in active_sessions:
                active_sessions[session_id]["status"] = "error"
                active_sessions[session_id]["error"] = str(e)
            raise
        finally:
            # Schedule cleanup
            asyncio.create_task(tts_manager.cleanup_session(session_id))
    
    return StreamingResponse(
        optimized_stream(),
        media_type="audio/wav",
        headers={"X-Session-ID": session_id}
    )

@router.post("/tts_stream_batch")
async def tts_stream_batch(
    request: TTSInputs, 
    options: StreamingOptions = StreamingOptions(),
    tts_manager: TTSManager = Depends(get_tts_manager),
    voice_storage: VoiceStorage = Depends(get_voice_storage),
    speaker_cache: SpeakerCache = Depends(get_speaker_cache)
):
    """
    Batch processing for long texts with streaming
    
    Features:
    - Splitting long texts into chunks
    - Parallel chunk processing
    - Generation progress tracking
    """
    session_id = generate_session_id()
    
    # Get speaker data
    gpt_cond_latent, speaker_embedding = await get_speaker_data(
        tts_manager.model,
        speaker_cache,
        voice_storage,
        speaker_id=request.speaker_id,
        audio_file_path=request.audio_file_path
    )
    
    # Process text into smaller chunks
    text_chunks = preprocess_text(request.text)
    
    async def batch_stream_generator():
        """Generate audio chunks from batch text processing"""
        active_sessions[session_id] = {
            "status": "processing",
            "text": request.text,
            "language": request.language,
            "created_at": asyncio.get_event_loop().time(),
            "progress": 0,
            "total_chunks": len(text_chunks)
        }
        
        # Special handling for first chunk with WAV header
        if options.add_wav_header:
            yield encode_audio_common(b"", encode_base64=False)
        
        # Process each text chunk
        for i, chunk_text in enumerate(text_chunks):
            # Update progress in session info
            if session_id in active_sessions:
                active_sessions[session_id]["progress"] = i
            
            # Process this chunk with model
            async with tts_manager.semaphore:
                out_chunks = tts_manager.model.inference_stream(
                    chunk_text,
                    request.language,
                    gpt_cond_latent,
                    speaker_embedding,
                    stream_chunk_size=options.stream_chunk_size,
                    enable_text_splitting=True,
                    generation_config=OPTIMIZED_GENERATION_CONFIG
                )
            
            # Pass generated audio chunks
            for audio_chunk in out_chunks:
                processed_chunk = tts_manager.postprocess(audio_chunk)
                yield processed_chunk.tobytes()
                
                # Check if session was canceled
                if session_id in active_sessions and active_sessions[session_id]["status"] == "canceled":
                    return
        
        # Update session status on completion
        if session_id in active_sessions:
            active_sessions[session_id]["status"] = "completed"
            active_sessions[session_id]["progress"] = len(text_chunks)
    
    return StreamingResponse(
        batch_stream_generator(),
        media_type="audio/wav",
        headers={"X-Session-ID": session_id}
    )