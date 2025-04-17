import time
from typing import Dict, Any, Tuple, Optional, TypeVar, Generic

T = TypeVar('T')

class SpeakerCache(Generic[T]):
    """Voice cache with performance metrics"""
    
    def __init__(self, ttl: int = 3600, max_size: int = 100):
        """
        Initialize cache with parameters
        
        Args:
            ttl: Record lifetime in seconds
            max_size: Maximum number of records in the cache
        """
        self.cache: Dict[str, Tuple[T, float]] = {}
        self.ttl = ttl
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # Cache cleanup interval in seconds
    
    def get(self, key: str) -> Optional[T]:
        """
        Get data from cache with TTL check
        
        Args:
            key: Key to search in the cache
            
        Returns:
            Data from cache or None if data not found or expired
        """
        now = time.time()
        
        # Periodic cleanup of expired records
        if now - self.last_cleanup > self.cleanup_interval:
            self._cleanup_expired()
        
        if key in self.cache:
            data, timestamp = self.cache[key]
            if now - timestamp < self.ttl:
                # Cache hit
                self.hits += 1
                # Update access time (for LRU)
                self.cache[key] = (data, now)
                return data
            # Expired
            del self.cache[key]
        
        # Cache miss
        self.misses += 1
        return None
    
    def set(self, key: str, data: T) -> None:
        """
        Add data to cache
        
        Args:
            key: Key to save in cache
            data: Data to save
        """
        now = time.time()
        
        # If cache is full, remove least recently used items
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._cleanup_lru()
        
        self.cache[key] = (data, now)
    
    def _cleanup_expired(self) -> None:
        """Clean up expired cache records"""
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
        """Remove least recently used cache items"""
        if not self.cache:
            return
        
        # Sort by last access time
        sorted_keys = sorted(self.cache.keys(), key=lambda k: self.cache[k][1])
        
        # Remove 25% of least recently used items
        keys_to_remove = sorted_keys[:max(1, len(sorted_keys) // 4)]
        
        for key in keys_to_remove:
            del self.cache[key]
        
        print(f"Cache LRU cleanup: removed {len(keys_to_remove)} least recently used entries. Current size: {len(self.cache)}", flush=True)
    
    def clear(self) -> None:
        """Complete cache cleanup"""
        self.cache.clear()
        print("Cache cleared", flush=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache operation statistics
        
        Returns:
            Dictionary with cache statistics
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