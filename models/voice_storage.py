import json
import time
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

class VoiceStorage:
    """Класс для работы с хранилищем голосов"""
    
    def __init__(self, storage_dir: Path):
        """
        Инициализация хранилища голосов
        
        Args:
            storage_dir: Директория для хранения файлов голосов
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.metadata_file = self.storage_dir / "metadata.json"
        
        # Загружаем метаданные или создаем новые
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def save_voice(self, 
                   voice_id: str, 
                   gpt_cond_latent: Union[torch.Tensor, List], 
                   speaker_embedding: Union[torch.Tensor, List], 
                   name: Optional[str] = None, 
                   description: Optional[str] = None) -> str:
        """
        Сохраняет голос в хранилище с правильной обработкой тензоров
        
        Args:
            voice_id: Идентификатор голоса
            gpt_cond_latent: Латентное представление для GPT-модели
            speaker_embedding: Векторное представление голоса
            name: Название голоса (опционально)
            description: Описание голоса (опционально)
            
        Returns:
            ID сохраненного голоса
        """
        # Проверяем и нормализуем размерность gpt_cond_latent
        if hasattr(gpt_cond_latent, "dim") and gpt_cond_latent.dim() < 3:
            # Добавляем размерности, если необходимо
            if gpt_cond_latent.dim() == 1:
                gpt_cond_latent = gpt_cond_latent.unsqueeze(0).unsqueeze(0)
            elif gpt_cond_latent.dim() == 2:
                gpt_cond_latent = gpt_cond_latent.unsqueeze(0)
        
        # Преобразуем тензоры в списки для сохранения в JSON
        if hasattr(gpt_cond_latent, "cpu"):
            # Убеждаемся, что тензор имеет правильную форму перед сохранением
            gpt_cond_latent_cpu = gpt_cond_latent.cpu()
            
            # Сохраняем форму тензора для последующего восстановления
            tensor_shape = list(gpt_cond_latent_cpu.shape)
            
            # Сохраняем данные как плоский список
            gpt_cond_latent_data = gpt_cond_latent_cpu.reshape(-1).tolist()
        else:
            # Если это не тензор, а список, пытаемся определить его форму
            if isinstance(gpt_cond_latent, list):
                if isinstance(gpt_cond_latent[0], list):
                    if isinstance(gpt_cond_latent[0][0], list):
                        # [1, n, d]
                        tensor_shape = [len(gpt_cond_latent), len(gpt_cond_latent[0]), len(gpt_cond_latent[0][0])]
                        gpt_cond_latent_data = [item for sublist1 in gpt_cond_latent for sublist2 in sublist1 for item in sublist2]
                    else:
                        # [n, d]
                        tensor_shape = [1, len(gpt_cond_latent), len(gpt_cond_latent[0])]
                        gpt_cond_latent_data = [item for sublist in gpt_cond_latent for item in sublist]
                else:
                    # [d]
                    tensor_shape = [1, 1, len(gpt_cond_latent)]
                    gpt_cond_latent_data = gpt_cond_latent
            else:
                raise ValueError("gpt_cond_latent must be a tensor or a list")
        
        # Обрабатываем speaker_embedding
        if hasattr(speaker_embedding, "cpu"):
            speaker_embedding_data = speaker_embedding.cpu().squeeze().tolist()
        else:
            speaker_embedding_data = speaker_embedding
        
        # Создаем файл для голоса
        voice_file = self.storage_dir / f"{voice_id}.json"
        voice_data = {
            "gpt_cond_latent": gpt_cond_latent_data,
            "gpt_cond_latent_shape": tensor_shape,  # Сохраняем форму тензора
            "speaker_embedding": speaker_embedding_data
        }
        
        with open(voice_file, "w") as f:
            json.dump(voice_data, f)
        
        # Обновляем метаданные
        self.metadata[voice_id] = {
            "name": name or voice_id,
            "description": description,
            "created_at": time.time(),
            "tensor_shape": tensor_shape  # Сохраняем форму тензора в метаданных
        }
        
        # Сохраняем обновленные метаданные
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f)
        
        return voice_id
    
    def get_voice(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """
        Получает голос из хранилища и восстанавливает правильную форму тензоров
        
        Args:
            voice_id: Идентификатор голоса
            
        Returns:
            Данные голоса или None, если голос не найден
        """
        voice_file = self.storage_dir / f"{voice_id}.json"
        
        if not voice_file.exists():
            return None
        
        with open(voice_file, "r") as f:
            voice_data = json.load(f)
        
        # Проверяем, есть ли информация о форме тензора
        if "gpt_cond_latent_shape" in voice_data:
            # Восстанавливаем форму тензора
            tensor_shape = voice_data["gpt_cond_latent_shape"]
            
            # Преобразуем плоский список обратно в нужную форму
            if len(tensor_shape) == 3:
                # Создаем вложенные списки для восстановления исходной формы
                latent_data = voice_data["gpt_cond_latent"]
                
                # Вычисляем размеры для реструктуризации
                batch_size, seq_len, dim = tensor_shape
                total_size = batch_size * seq_len * dim
                
                # Проверяем согласованность данных и формы
                if len(latent_data) != total_size:
                    print(f"Warning: Data size mismatch for voice {voice_id}. Expected {total_size}, got {len(latent_data)}. Using original data.", flush=True)
                    return voice_data
                
                # Возвращаем данные в исходной структуре
                restructured_data = {
                    "gpt_cond_latent": latent_data,  # Оставляем плоским, torch.tensor восстановит форму
                    "gpt_cond_latent_shape": tensor_shape,
                    "speaker_embedding": voice_data["speaker_embedding"]
                }
                return restructured_data
            else:
                # Если форма некорректна, возвращаем исходные данные
                return voice_data
        else:
            # Если нет информации о форме, возвращаем исходные данные
            return voice_data
    
    def list_voices(self) -> Dict[str, Dict[str, Any]]:
        """
        Возвращает список всех сохраненных голосов с метаданными
        
        Returns:
            Словарь с метаданными голосов
        """
        return self.metadata
    
    def delete_voice(self, voice_id: str) -> bool:
        """
        Удаляет голос из хранилища
        
        Args:
            voice_id: Идентификатор голоса
            
        Returns:
            True если голос был удален, False если голос не найден
        """
        voice_file = self.storage_dir / f"{voice_id}.json"
        
        if voice_file.exists():
            voice_file.unlink()
        
        if voice_id in self.metadata:
            del self.metadata[voice_id]
            
            # Сохраняем обновленные метаданные
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f)
            
            return True
        
        return False
    
    def has_voice(self, voice_id: str) -> bool:
        """
        Проверяет наличие голоса в хранилище
        
        Args:
            voice_id: Идентификатор голоса
            
        Returns:
            True если голос существует, False если нет
        """
        voice_file = self.storage_dir / f"{voice_id}.json"
        return voice_file.exists() and voice_id in self.metadata