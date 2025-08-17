"""
Intent Classification Cache
LRU cache with TTL for storing LLM intent classification results.
"""

import time
import hashlib
from typing import Dict, Optional, Any
from collections import OrderedDict


class IntentCache:
    """LRU cache with TTL for intent classification results."""
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        """Initialize the cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl_hours: Time-to-live in hours
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    
    def get(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached result for a query hash.
        
        Args:
            query_hash: Hash of the query
            
        Returns:
            Cached result if valid, None otherwise
        """
        if query_hash not in self.cache:
            return None
        
        # Check TTL
        cached_item = self.cache[query_hash]
        if time.time() - cached_item["timestamp"] > self.ttl_seconds:
            # Expired, remove from cache
            del self.cache[query_hash]
            return None
        
        # Move to end (LRU)
        self.cache.move_to_end(query_hash)
        return cached_item["result"]
    
    def set(self, query_hash: str, result: Dict[str, Any], 
            model_version: Optional[str] = None) -> None:
        """Cache a result for a query hash.
        
        Args:
            query_hash: Hash of the query
            result: Classification result to cache
            model_version: Version of the model used
        """
        # Remove if exists (will be re-added at end)
        if query_hash in self.cache:
            del self.cache[query_hash]
        
        # Add new item
        self.cache[query_hash] = {
            "result": result,
            "timestamp": time.time(),
            "model_version": model_version
        }
        
        # Enforce max size
        if len(self.cache) > self.max_size:
            # Remove oldest item
            self.cache.popitem(last=False)
    
    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size.
        
        Returns:
            Number of cached items
        """
        return len(self.cache)
    
    def cleanup_expired(self) -> int:
        """Remove expired items from cache.
        
        Returns:
            Number of items removed
        """
        current_time = time.time()
        expired_keys = [
            key for key, item in self.cache.items()
            if current_time - item["timestamp"] > self.ttl_seconds
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        current_time = time.time()
        expired_count = sum(
            1 for item in self.cache.values()
            if current_time - item["timestamp"] > self.ttl_seconds
        )
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "expired_items": expired_count,
            "ttl_hours": self.ttl_seconds / 3600
        }


def hash_query(query: str) -> str:
    """Generate a hash for a query string.
    
    Args:
        query: Query string to hash
        
    Returns:
        SHA256 hash of the normalized query
    """
    # Normalize query for consistent hashing
    normalized = query.lower().strip()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16] 