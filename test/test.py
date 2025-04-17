import requests
import os
import pyaudio
import time

def test_tts_stream():
    # Endpoint URL
    url = "http://localhost:8000/tts_stream"
    
    # Test data
    data = {
        "request": {
            "text": "Yes! I can repeat myself in my own voice, awesome!",
            "language": "en",
            "audio_file_path": os.path.join(os.path.dirname(os.path.abspath(__file__)),"test.mp3")
        },
        "options": {
            "stream_chunk_size": 20
        }
    }
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    try:
        # Start timing the request
        start_time = time.time()
        
        # Send POST request with streaming response
        response = requests.post(url, json=data, stream=True)
        
        # Check if request was successful
        assert response.status_code == 200
        
        # Check if response is streaming
        assert response.headers.get('content-type') == 'audio/wav'
        
        # Open stream for playback
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=22050,
            output=True
        )
        
        first_byte_received = False
        first_byte_time = None
        total_bytes = 0
        
        # Play audio as chunks are received
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                if not first_byte_received:
                    first_byte_time = time.time()
                    first_byte_received = True
                    print(f"\nTime to first byte: {(first_byte_time - start_time)*1000:.2f} ms")
                
                total_bytes += len(chunk)
                stream.write(chunk)
        
        # Print statistics
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nTotal playback time: {total_time:.2f} sec")
        print(f"Average transfer rate: {total_bytes / total_time / 1024:.2f} KB/sec")
        
        # Close stream
        stream.stop_stream()
        stream.close()
        
    finally:
        # Always close PyAudio
        p.terminate()

if __name__ == "__main__":
    test_tts_stream()
