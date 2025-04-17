import os
import sys
import time
import uvicorn
import warnings

# Логирование начала запуска
print("\n" + "=" * 80)
print(f"XTTS Server Startup - {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

print("[1/4] Активация трассировки памяти...")
import tracemalloc
tracemalloc.start()
print("      ✅ Трассировка памяти активирована")

# Подавление предупреждений torch
print("[2/4] Подавление несущественных предупреждений...")
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")
warnings.filterwarnings("ignore", message=".*weights_only=False.*")
print("      ✅ Предупреждения подавлены")

# Проверка зависимостей
print("[3/4] Проверка зависимостей...")
try:
    import torch
    gpu_available = torch.cuda.is_available()
    print(f"      ✅ PyTorch: {torch.__version__} (GPU {'доступен 🚀' if gpu_available else 'не доступен 🐢'})")
    
    import fastapi
    print(f"      ✅ FastAPI: {fastapi.__version__}")
    
    from TTS.tts.models.xtts import Xtts
    print("      ✅ XTTS модель доступна")
except ImportError as e:
    print(f"     ❌ Ошибка: {str(e)}")
    print("\nПожалуйста, установите необходимые зависимости:")
    print("pip install -r requirements.txt")
    sys.exit(1)

# Настройка и конфигурация
print("[4/4] Чтение конфигурации...")
# Получаем порт из переменных окружения или используем порт по умолчанию
port = int(os.environ.get("PORT", 8000))
host = os.environ.get("HOST", "0.0.0.0")
reload_mode = os.environ.get("RELOAD", "true").lower() in ("true", "1", "t")
workers = int(os.environ.get("WORKERS", 1))
log_level = os.environ.get("LOG_LEVEL", "info")

print(f"      ✅ Хост: {host}")
print(f"      ✅ Порт: {port}")
print(f"      ✅ Автоперезагрузка: {'включена' if reload_mode else 'выключена'}")
print(f"      ✅ Количество рабочих процессов: {workers}")
print(f"      ✅ Уровень логирования: {log_level}")

# Запуск сервера
print("[5/5] Запуск XTTS сервера...")
print("\nСервер запускается. Это может занять некоторое время...")
print("Для прерывания нажмите Ctrl+C\n")

if __name__ == "__main__":
    # Запускаем сервер
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