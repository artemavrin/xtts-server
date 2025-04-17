# XTTS Streaming Server

Optimized server for streaming synthesized speech using the XTTS model with support for voice cloning and caching.

## Features

- **Speech Streaming**: Instant synthesis start with audio streaming as it's generated
- **Voice Cloning**: Create voices from audio files with reusability
- **Voice Storage**: Save cloned voices for later use
- **Caching**: Performance optimization through voice caching in memory
- **Batch Processing**: Efficient processing of long texts by splitting into chunks
- **Monitoring**: Track generation status and manage sessions
- **Multilingual**: Support for all languages available in XTTS

## Architecture

The project has a modular architecture:

```
xtts-server/
├── main.py              # Main file with FastAPI application
├── config.py            # Settings and configurations
├── models/              # Data models
│   ├── __init__.py
│   ├── inputs.py        # Input data models (Pydantic)
│   └── voice_storage.py # Voice storage class
├── services/            # Services
│   ├── __init__.py
│   ├── audio.py         # Audio processing functions
│   ├── cache.py         # Voice caching
│   ├── tts.py           # Main TTS logic
│   └── utils.py         # Utility functions
├── api/                 # API layer
│   ├── __init__.py
│   ├── routes.py        # API routes
│   └── responses.py     # Response handling
├── requirements.txt
└── run.py               # Startup script
```

## Installation

### Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/xtts-streaming-server.git
cd xtts-streaming-server
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Starting the Server

Start the server with:
```bash
python run.py
```

The server will be available at `http://localhost:8000`. Swagger documentation is available at the same address.

## Environment Variables Configuration

The server can be configured using environment variables:

- `PORT` - Port to run on (default 8000)
- `DEBUG` - Debug mode (default False)
- `NUM_THREADS` - Number of CPU threads (default number of cores)
- `MAX_CONCURRENT_REQUESTS` - Maximum number of concurrent requests (default 10)
- `CUSTOM_MODEL_PATH` - Path to custom XTTS model
- `USE_CPU` - Force CPU usage even with GPU available (default False)
- `SPEAKER_CACHE_TTL` - Voice cache lifetime in seconds (default 3600)
- `SPEAKER_CACHE_MAX_SIZE` - Maximum voice cache size (default 100)
- `PRELOAD_VOICES` - List of voices to preload (comma-separated)
- `VOICES_DIR` - Directory for storing saved voices (default ./saved_voices)

## API Documentation

### Main Endpoints

- **POST /tts_stream** - Stream speech synthesis
- **POST /tts** - Full audio synthesis (non-streaming)
- **POST /clone_speaker** - Clone voice from audio file
- **POST /save_voice/{voice_id}** - Save cloned voice
- **GET /saved_voices** - Get list of saved voices
- **DELETE /saved_voices/{voice_id}** - Delete saved voice
- **GET /cache/stats** - Voice cache statistics
- **POST /cache/clear** - Clear voice cache
- **GET /stream_status/{session_id}** - Streaming session status
- **DELETE /stream_cancel/{session_id}** - Cancel streaming session
- **GET /available_speakers** - Available built-in voices
- **GET /languages** - Available languages
- **GET /health** - Server health check

## Usage Examples

### Speech Synthesis with Cloned Voice

1. Clone voice:
```bash
curl -X POST "http://localhost:8000/clone_speaker" \
  -F "wav_file=@path/to/voice-sample.wav" \
  -F 'options={"response_format": "save", "save_id": "my-voice", "name": "My Voice", "description": "Voice description"}'
```

2. Synthesize speech with saved voice:
```bash
curl -X POST "http://localhost:8000/tts_stream" \
  -H "Content-Type: application/json" \
  -d '{"text": "Text to synthesize", "language": "en", "speaker_id": "my-voice"}'
```

### Get Cache Statistics

```bash
curl -X GET "http://localhost:8000/cache/stats"
```

### Check Session Status

```bash
curl -X GET "http://localhost:8000/stream_status/{session_id}"
```

## Performance Optimizations

The server includes several optimizations for maximum performance:

1. **Voice Caching**: Voices are cached in memory for quick access
2. **Voice Preloading**: Frequently used voices are loaded at server startup
3. **Efficient Resource Management**: Semaphores for model access control
4. **Batch Processing**: Splitting long texts into chunks for efficient processing
5. **Minimal Latency**: Immediate WAV header and silence sending for connection establishment

## Troubleshooting

### Tensor Dimension Error

If you see an error like `Tensors must have same number of dimensions: got 2 and 3`, this means there's an issue with voice tensor dimensions. Check that saved voices have the correct shape.

### Memory Leaks

If you observe memory leaks, add the `SPEAKER_CACHE_TTL` environment variable with a smaller value for more frequent cache cleanup.

## License

This project is distributed under the MIT license.