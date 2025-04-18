import os
import sys
import time
import uvicorn
import warnings

# Log startup sequence
print("\n" + "=" * 80)
print(f"XTTS Server Startup - {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

print("[1/5] Activating memory tracing...")
import tracemalloc
tracemalloc.start()
print("      ✅ Memory tracing activated")

# Suppress torch warnings
print("[2/5] Suppressing non-essential warnings...")
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")
warnings.filterwarnings("ignore", message=".*weights_only=False.*")
print("      ✅ Warnings suppressed")

# Check dependencies
print("[3/5] Checking dependencies...")
try:
    import torch
    gpu_available = torch.cuda.is_available()
    mps_available = torch.mps.is_available()
    print(f"      ✅ PyTorch: {torch.__version__}")
    if gpu_available:
        print(f"      ✅ GPU: available 🚀")
    if mps_available:
        print(f"      ✅ MPS: available 🚀")
    else:
        print(f"      ❌ no GPU or MPS available")

    import transformers
    print(f"      ✅ Transformers: {transformers.__version__}")

    import fastapi
    print(f"      ✅ FastAPI: {fastapi.__version__}")
    
    from TTS.tts.models.xtts import Xtts
    print("      ✅ XTTS model available")
except ImportError as e:
    print(f"     ❌ Error: {str(e)}")
    print("\nPlease install the required dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)

# Setup and configuration
print("[4/5] Reading configuration...")
# Get port from environment variables or use default port
port = int(os.environ.get("PORT", 8000))
host = os.environ.get("HOST", "0.0.0.0")
reload_mode = os.environ.get("RELOAD", "true").lower() in ("true", "1", "t")
workers = int(os.environ.get("WORKERS", 1))
log_level = os.environ.get("LOG_LEVEL", "info")

print(f"      ✅ Host: {host}")
print(f"      ✅ Port: {port}")
print(f"      ✅ Auto-reload: {'enabled' if reload_mode else 'disabled'}")
print(f"      ✅ Number of workers: {workers}")
print(f"      ✅ Log level: {log_level}")

# Start the server
print("[5/5] Starting XTTS server...")
print("\nServer is starting. This may take some time...")
print("Press Ctrl+C to interrupt\n")

if __name__ == "__main__":
    # Run the server
    config = uvicorn.Config(
        "main:app",
        host=host,
        port=port,
        reload=reload_mode,
        workers=workers,
        log_level=log_level
    )
    server = uvicorn.Server(config)
    server.run()