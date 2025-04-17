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
    Постобработка выходной волновой формы
    
    Args:
        wav: Тензор или список тензоров с аудиоданными
        
    Returns:
        Обработанные аудиоданные как numpy-массив
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
    Кодирует аудиоданные в формат WAV и опционально в base64
    
    Args:
        frame_input: Входные аудиоданные в байтах
        encode_base64: Флаг для кодирования результата в base64
        sample_rate: Частота дискретизации
        sample_width: Глубина сэмпла в байтах
        channels: Количество каналов
        
    Returns:
        Закодированные аудиоданные (строка base64 или байты)
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
    Генерирует аудио-тишину заданной длительности
    
    Args:
        duration_ms: Длительность в миллисекундах
        sample_rate: Частота дискретизации
        
    Returns:
        Аудиоданные тишины в байтах
    """
    num_samples = int(duration_ms * sample_rate / 1000)
    silence = np.zeros(num_samples, dtype=np.int16)
    return silence.tobytes()

def preprocess_text(text: str, max_chunk_length: int = MAX_TEXT_CHUNK_SIZE) -> List[str]:
    """
    Предобработка текста для более эффективной потоковой передачи
    
    Разделение длинных параграфов на меньшие, управляемые фрагменты, которые
    могут быть более эффективно обработаны TTS-моделью.
    
    Args:
        text: Исходный текст
        max_chunk_length: Максимальная длина текстового фрагмента
        
    Returns:
        Список текстовых фрагментов
    """
    # Разделение текста по границам предложений
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Группировка предложений в фрагменты разумного размера
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Если добавление этого предложения сделает фрагмент слишком длинным, начинаем новый фрагмент
        if len(current_chunk) + len(sentence) > max_chunk_length:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Добавляем последний фрагмент, если он не пустой
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def ensure_tensor_dimensions(tensor: Union[torch.Tensor, List]) -> torch.Tensor:
    """
    Проверяет и исправляет размерность тензора для совместимости с моделью XTTS
    
    GPT-кондиционирующий латентный тензор должен иметь размерность [1, n, d],
    где n - количество векторов, d - размер каждого вектора.
    
    Args:
        tensor: Входной тензор или список
        
    Returns:
        Тензор с правильной размерностью
    """
    from config import device
    
    # Проверяем, является ли входной объект тензором
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor).to(device)
    
    # Получаем текущую размерность тензора
    dims = tensor.dim()
    
    # Исправляем размерность, если необходимо
    if dims == 1:
        # [d] -> [1, 1, d]
        return tensor.unsqueeze(0).unsqueeze(0)
    elif dims == 2:
        # [n, d] -> [1, n, d]
        return tensor.unsqueeze(0)
    elif dims == 3:
        # [1, n, d] - правильная размерность
        return tensor
    else:
        # Неожиданная размерность
        print(f"Warning: Unexpected tensor dimensions: {tensor.size()}", flush=True)
        # Пытаемся исправить, если возможно
        if dims > 3:
            return tensor.squeeze(0) if tensor.size(0) == 1 else tensor[:1]
        return tensor

def ensure_speaker_embedding_dimensions(speaker_embedding: torch.Tensor) -> torch.Tensor:
    """
    Исправляет размерность speaker_embedding для использования в модели
    
    Args:
        speaker_embedding: Тензор с вложением голоса
        
    Returns:
        Тензор с правильной размерностью [1, 512]
    """
    if speaker_embedding.dim() == 1:
        return speaker_embedding.unsqueeze(0)  # [512] -> [1, 512]
    return speaker_embedding

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Нормализует аудиоданные к диапазону [-1, 1]
    
    Args:
        audio: Аудиоданные как numpy-массив
        
    Returns:
        Нормализованные аудиоданные
    """
    return audio / np.max(np.abs(audio))

def resample_audio(
    audio: np.ndarray,
    original_rate: int,
    target_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Передискретизирует аудиоданные до целевой частоты дискретизации
    
    Args:
        audio: Аудиоданные как numpy-массив
        original_rate: Исходная частота дискретизации
        target_rate: Целевая частота дискретизации
        
    Returns:
        Передискретизированные аудиоданные
    """
    if original_rate == target_rate:
        return audio
    
    # Вычисляем коэффициент передискретизации
    ratio = target_rate / original_rate
    
    # Вычисляем новую длину
    new_length = int(len(audio) * ratio)
    
    # Создаем временную шкалу для интерполяции
    old_indices = np.arange(len(audio))
    new_indices = np.linspace(0, len(audio) - 1, new_length)
    
    # Линейная интерполяция
    return np.interp(new_indices, old_indices, audio)

def mix_audio(audio1: np.ndarray, audio2: np.ndarray, ratio: float = 0.5) -> np.ndarray:
    """
    Смешивает два аудиосигнала с заданным соотношением
    
    Args:
        audio1: Первый аудиосигнал
        audio2: Второй аудиосигнал
        ratio: Соотношение смешивания (0.0 - только audio1, 1.0 - только audio2)
        
    Returns:
        Смешанный аудиосигнал
    """
    # Нормализуем сигналы
    audio1 = normalize_audio(audio1)
    audio2 = normalize_audio(audio2)
    
    # Смешиваем с заданным соотношением
    mixed = audio1 * (1 - ratio) + audio2 * ratio
    
    # Нормализуем результат
    return normalize_audio(mixed)

def apply_fade(audio: np.ndarray, fade_length: int = 100) -> np.ndarray:
    """
    Применяет плавное затухание к началу и концу аудиосигнала
    
    Args:
        audio: Аудиосигнал
        fade_length: Длина затухания в сэмплах
        
    Returns:
        Аудиосигнал с примененным затуханием
    """
    # Создаем маску затухания
    fade_in = np.linspace(0, 1, fade_length)
    fade_out = np.linspace(1, 0, fade_length)
    
    # Применяем затухание
    audio[:fade_length] *= fade_in
    audio[-fade_length:] *= fade_out
    
    return audio