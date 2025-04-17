import time
from typing import Dict, Any, Tuple, Optional, TypeVar, Generic

T = TypeVar('T')

class SpeakerCache(Generic[T]):
    """Кэш голосов с метриками производительности"""
    
    def __init__(self, ttl: int = 3600, max_size: int = 100):
        """
        Инициализация кэша с параметрами
        
        Args:
            ttl: Время жизни записей в секундах
            max_size: Максимальное количество записей в кэше
        """
        self.cache: Dict[str, Tuple[T, float]] = {}
        self.ttl = ttl
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # Интервал очистки кэша в секундах
    
    def get(self, key: str) -> Optional[T]:
        """
        Получение данных из кэша с проверкой TTL
        
        Args:
            key: Ключ для поиска в кэше
            
        Returns:
            Данные из кэша или None, если данные не найдены или устарели
        """
        now = time.time()
        
        # Периодическая очистка устаревших записей
        if now - self.last_cleanup > self.cleanup_interval:
            self._cleanup_expired()
        
        if key in self.cache:
            data, timestamp = self.cache[key]
            if now - timestamp < self.ttl:
                # Кэш-хит
                self.hits += 1
                # Обновляем время доступа (для LRU)
                self.cache[key] = (data, now)
                return data
            # Истекло
            del self.cache[key]
        
        # Кэш-мисс
        self.misses += 1
        return None
    
    def set(self, key: str, data: T) -> None:
        """
        Добавление данных в кэш
        
        Args:
            key: Ключ для сохранения в кэше
            data: Данные для сохранения
        """
        now = time.time()
        
        # Если кэш заполнен, удаляем наименее недавно использованные элементы
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._cleanup_lru()
        
        self.cache[key] = (data, now)
    
    def _cleanup_expired(self) -> None:
        """Очистка устаревших записей кэша"""
        now = time.time()
        expired_keys = []
        
        for key, (_, timestamp) in self.cache.items():
            if now - timestamp > self.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        self.last_cleanup = now
        
        if expired_keys:
            print(f"Cache cleanup: removed {len(expired_keys)} expired entries. Current size: {len(self.cache)}", flush=True)
    
    def _cleanup_lru(self) -> None:
        """Удаление наименее недавно использованных элементов кэша"""
        if not self.cache:
            return
        
        # Сортировка по времени последнего доступа
        sorted_keys = sorted(self.cache.keys(), key=lambda k: self.cache[k][1])
        
        # Удаляем 25% наименее недавно использованных элементов
        keys_to_remove = sorted_keys[:max(1, len(sorted_keys) // 4)]
        
        for key in keys_to_remove:
            del self.cache[key]
        
        print(f"Cache LRU cleanup: removed {len(keys_to_remove)} least recently used entries. Current size: {len(self.cache)}", flush=True)
    
    def clear(self) -> None:
        """Полная очистка кэша"""
        self.cache.clear()
        print("Cache cleared", flush=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики работы кэша
        
        Returns:
            Словарь со статистикой кэша
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": hit_rate,
            "ttl_seconds": self.ttl
        }