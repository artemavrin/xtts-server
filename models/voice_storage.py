import json
import time
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
from datetime import datetime

class VoiceStorage:
    """Voice storage with metadata"""
    
    def __init__(self, storage_dir: str = "voices"):
        """Initialize voice storage
        
        Args:
            storage_dir: Directory for storing voices
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata from file or create new if not exists"""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        else:
            return {}
    
    def save_voice(self, 
                   voice_id: str, 
                   gpt_cond_latent: Union[torch.Tensor, List], 
                   speaker_embedding: Union[torch.Tensor, List], 
                   name: Optional[str] = None, 
                   description: Optional[str] = None) -> str:
        """
        Save voice to storage with proper tensor handling
        
        Args:
            voice_id: Voice identifier
            gpt_cond_latent: GPT model latent representation
            speaker_embedding: Voice embedding vector
            name: Voice name (optional)
            description: Voice description (optional)
            
        Returns:
            ID of saved voice
        """
        # Check and normalize gpt_cond_latent dimensions
        if hasattr(gpt_cond_latent, "dim") and gpt_cond_latent.dim() < 3:
            # Add dimensions if needed
            if gpt_cond_latent.dim() == 1:
                gpt_cond_latent = gpt_cond_latent.unsqueeze(0).unsqueeze(0)
            elif gpt_cond_latent.dim() == 2:
                gpt_cond_latent = gpt_cond_latent.unsqueeze(0)
        
        # Convert tensors to lists for JSON storage
        if hasattr(gpt_cond_latent, "cpu"):
            # Ensure tensor has correct shape before saving
            gpt_cond_latent_cpu = gpt_cond_latent.cpu()
            
            # Save tensor shape for later restoration
            tensor_shape = list(gpt_cond_latent_cpu.shape)
            
            # Save data as flat list
            gpt_cond_latent_data = gpt_cond_latent_cpu.reshape(-1).tolist()
        else:
            # If not a tensor but a list, try to determine its shape
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
        
        # Process speaker_embedding
        if hasattr(speaker_embedding, "cpu"):
            speaker_embedding_data = speaker_embedding.cpu().squeeze().tolist()
        else:
            speaker_embedding_data = speaker_embedding
        
        # Create voice file
        voice_file = self.storage_dir / f"{voice_id}.json"
        voice_data = {
            "gpt_cond_latent": gpt_cond_latent_data,
            "gpt_cond_latent_shape": tensor_shape,  # Save tensor shape
            "speaker_embedding": speaker_embedding_data
        }
        
        with open(voice_file, "w") as f:
            json.dump(voice_data, f)
        
        # Update metadata
        self.metadata[voice_id] = {
            "name": name or voice_id,
            "description": description,
            "created_at": time.time(),
            "tensor_shape": tensor_shape  # Save tensor shape in metadata
        }
        
        # Save updated metadata
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f)
        
        return voice_id
    
    def get_voice(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """
        Get voice from storage and restore correct tensor shapes
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            Voice data or None if voice not found
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

    def _save_metadata(self) -> None:
        """Save metadata to file"""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)