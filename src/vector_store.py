"""Vector Store - Endee operations using correct API"""
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from endee import Endee
from src.config import Config

@dataclass
class SearchResult:
    id: str
    content: str
    source: str
    score: float
    metadata: Dict[str, Any]

class VectorStore:
    def __init__(self):
        from src.embedding_engine import EmbeddingEngine
        self.embedding_engine = EmbeddingEngine()
        self.client = Endee()
        self.index_name = Config.COLLECTION_NAME
        self._setup_index()

    def _setup_index(self):
        try:
            existing = self.client.list_indexes()
            names = [i if isinstance(i, str) else i.get("name", "") for i in (existing or [])]
            if self.index_name not in names:
                self.client.create_index(
                    name=self.index_name,
                    dimension=Config.EMBEDDING_DIMENSION,
                    space_type="cosine"
                )
                print(f"✅ Created index: {self.index_name}")
            else:
                print(f"✅ Using existing index: {self.index_name}")
        except Exception as e:
            print(f"⚠️ Index setup warning: {e}")

    def add_documents(self, chunks: List[Dict]) -> int:
        try:
            vectors = []
            for chunk in chunks:
                embedding = self.embedding_engine.embed_text(chunk["content"])
                vectors.append({
                    "id": chunk["id"],
                    "vector": embedding,
                    "metadata": {
                        "content": chunk["content"],
                        "source": chunk.get("source", ""),
                        "chunk_index": chunk.get("chunk_index", 0)
                    }
                })
            self.client.upsert(index_name=self.index_name, vectors=vectors)
            return len(vectors)
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            return 0

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        try:
            query_vector = self.embedding_engine.embed_text(query)
            results = self.client.search(
                index_name=self.index_name,
                vector=query_vector,
                top_k=top_k
            )
            search_results = []
            for r in (results or []):
                meta = r.get("metadata", {})
                search_results.append(SearchResult(
                    id=r.get("id", ""),
                    content=meta.get("content", ""),
                    source=meta.get("source", ""),
                    score=r.get("score", 0.0),
                    metadata=meta
                ))
            return search_results
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []

    def get_store_stats(self) -> Dict:
        try:
            info = self.client.get_index(self.index_name)
            return {
                "collection": self.index_name,
                "total_vectors": info.get("count", 0) if info else 0,
                "embedding_model": Config.EMBEDDING_MODEL,
                "dimension": Config.EMBEDDING_DIMENSION
            }
        except Exception as e:
            return {
                "collection": self.index_name,
                "total_vectors": 0,
                "embedding_model": Config.EMBEDDING_MODEL,
                "dimension": Config.EMBEDDING_DIMENSION
            }

    def delete_document(self, source: str):
        try:
            self.client.delete(
                index_name=self.index_name,
                filter={"source": source}
            )
        except Exception as e:
            print(f"⚠️ Delete warning: {e}")