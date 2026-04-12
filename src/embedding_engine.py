from sentence_transformers import SentenceTransformer
from typing import List, Union, Dict
import numpy as np
from functools import lru_cache
import hashlib
import json

class EmbeddingEngine:
    """Advanced embedding generation with caching"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding engine
        
        Args:
            model_name: Name of the sentence transformer model
        """
        print(f"🔄 Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.cache = {}
        print(f"✅ Model loaded! Embedding dimension: {self.dimension}")
    
    def encode(self, text: Union[str, List[str]], use_cache: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text
        
        Args:
            text: Single text or list of texts
            use_cache: Whether to use caching
            
        Returns:
            Embeddings as numpy array(s)
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        embeddings = []
        texts_to_encode = []
        cache_keys = []
        
        for t in texts:
            if use_cache:
                cache_key = self._get_cache_key(t)
                cache_keys.append(cache_key)
                
                if cache_key in self.cache:
                    embeddings.append(self.cache[cache_key])
                else:
                    texts_to_encode.append(t)
            else:
                texts_to_encode.append(t)
        
        # Encode uncached texts
        if texts_to_encode:
            new_embeddings = self.model.encode(texts_to_encode, convert_to_numpy=True)
            
            # Cache new embeddings
            if use_cache:
                for i, t in enumerate(texts_to_encode):
                    cache_key = self._get_cache_key(t)
                    self.cache[cache_key] = new_embeddings[i]
            
            embeddings.extend(new_embeddings)
        
        return embeddings[0] if is_single else embeddings
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Encode texts in batches for efficiency
        
        Args:
            texts: List of texts
            batch_size: Batch size for encoding
            
        Returns:
            List of embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.encode(batch, use_cache=True)
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
            "model_name": self.model._model_card_vars.get('model_name', 'unknown'),
            "dimension": self.dimension
        }
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.cache.clear()
        print("✅ Embedding cache cleared")