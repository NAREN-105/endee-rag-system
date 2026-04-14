"""Vector Store - Endee operations using correct SDK API"""
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

    @property
    def text(self):
        return self.content
        
    @property
    def doc_id(self):
        return self.id.split("_chunk_")[0] if "_chunk_" in self.id else self.id

class VectorStore:
    def __init__(self):
        from src.embedding_engine import EmbeddingEngine
        self.embedding_engine = EmbeddingEngine()
        self.client = Endee()
        self.index_name = Config.COLLECTION_NAME
        self._index = None  # Cached index object
        self._setup_index()

    def _setup_index(self):
        """Create or connect to the Endee index"""
        try:
            existing = self.client.list_indexes()
            names = []
            if existing:
                for i in existing:
                    if isinstance(i, str):
                        names.append(i)
                    elif isinstance(i, dict):
                        names.append(i.get("name", ""))
                    else:
                        # Could be an object with a name attribute
                        names.append(getattr(i, "name", str(i)))

            if self.index_name not in names:
                self.client.create_index(
                    name=self.index_name,
                    dimension=Config.EMBEDDING_DIMENSION,
                    space_type="cosine"
                )
                print(f"✅ Created index: {self.index_name}")
            else:
                print(f"✅ Using existing index: {self.index_name}")

            # Cache the index object for upsert/query operations
            self._index = self.client.get_index(name=self.index_name)
            print(f"✅ Connected to Endee index: {self.index_name}")

        except Exception as e:
            print(f"⚠️ Index setup warning: {e}")
            print(f"   Make sure Endee server is running on http://localhost:8080")
            self._index = None

    def _ensure_index(self):
        """Ensure we have a valid index reference"""
        if self._index is None:
            try:
                self._index = self.client.get_index(name=self.index_name)
            except Exception as e:
                print(f"❌ Cannot connect to Endee index: {e}")
                raise ConnectionError(
                    "Endee database not reachable. "
                    "Make sure Endee server is running: "
                    "docker run -p 8080:8080 -v ./endee-data:/data --name endee-server endeeio/endee-server:latest"
                )
        return self._index

    def add_documents(self, chunks: List[Dict]) -> int:
        """Insert document chunks into the Endee index using upsert"""
        try:
            index = self._ensure_index()
            vectors = []
            for chunk in chunks:
                embedding = self.embedding_engine.encode(chunk["content"])
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                vectors.append({
                    "id": chunk["id"],
                    "vector": embedding,
                    "meta": {
                        "content": chunk["content"],
                        "source": chunk.get("source", ""),
                        "chunk_index": chunk.get("chunk_index", 0)
                    }
                })

            # Endee SDK: upsert via index object, max 1000 per call
            batch_size = 1000
            total_inserted = 0
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                index.upsert(batch)
                total_inserted += len(batch)

            print(f"✅ Upserted {total_inserted} vectors into Endee")
            return total_inserted
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            return 0

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search the Endee index using query vector"""
        try:
            index = self._ensure_index()
            query_vector = self.embedding_engine.encode(query)
            if hasattr(query_vector, 'tolist'):
                query_vector = query_vector.tolist()

            # Endee SDK: query via index object
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                include_vectors=False
            )

            search_results = []
            for r in (results or []):
                # Endee returns dicts with 'id', 'similarity', 'meta'
                if isinstance(r, dict):
                    meta = r.get("meta", r.get("metadata", {}))
                    search_results.append(SearchResult(
                        id=r.get("id", ""),
                        content=meta.get("content", ""),
                        source=meta.get("source", ""),
                        score=r.get("similarity", r.get("score", 0.0)),
                        metadata=meta
                    ))
                else:
                    # Handle object-style results
                    meta = getattr(r, "meta", getattr(r, "metadata", {}))
                    if isinstance(meta, dict):
                        content = meta.get("content", "")
                        source = meta.get("source", "")
                    else:
                        content = ""
                        source = ""
                    search_results.append(SearchResult(
                        id=getattr(r, "id", ""),
                        content=content,
                        source=source,
                        score=getattr(r, "similarity", getattr(r, "score", 0.0)),
                        metadata=meta if isinstance(meta, dict) else {}
                    ))

            return search_results
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []

    def semantic_search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[SearchResult]:
        """Semantic search (delegates to search)"""
        return self.search(query, top_k)

    def multi_query_search(self, queries: List[str], top_k: int = 3) -> List[SearchResult]:
        """Search with multiple queries and merge results"""
        results = []
        seen_ids = set()
        for q in queries:
            for r in self.search(q, top_k):
                if r.id not in seen_ids:
                    seen_ids.add(r.id)
                    results.append(r)
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k * len(queries)]

    def filtered_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Filtered search (delegates to search)"""
        return self.search(query, top_k)

    def get_store_stats(self) -> Dict:
        """Get index statistics from Endee"""
        try:
            info = self.client.get_index(self.index_name)
            # info could be a dict or an object
            if isinstance(info, dict):
                count = info.get("count", info.get("vector_count", 0))
            else:
                count = getattr(info, "count", getattr(info, "vector_count", 0))
            return {
                "collection": self.index_name,
                "total_vectors": count if count else 0,
                "embedding_model": Config.EMBEDDING_MODEL,
                "dimension": Config.EMBEDDING_DIMENSION
            }
        except Exception as e:
            print(f"⚠️ Stats warning: {e}")
            return {
                "collection": self.index_name,
                "total_vectors": 0,
                "embedding_model": Config.EMBEDDING_MODEL,
                "dimension": Config.EMBEDDING_DIMENSION
            }

    def delete_document(self, source: str):
        """Delete vectors by source filter"""
        try:
            index = self._ensure_index()
            # Try using index-level delete with filter
            index.delete(
                filter={"source": {"$eq": source}}
            )
            print(f"✅ Deleted vectors for source: {source}")
        except Exception as e:
            print(f"⚠️ Delete warning: {e}")

    def get_all_sources(self) -> List[str]:
        """Get list of all unique source names (best-effort)"""
        try:
            # This is a best-effort — Endee doesn't have a direct 'list metadata' API
            # We do a broad search to find sources
            results = self.search("document", top_k=100)
            sources = list({r.source for r in results if r.source})
            return sources
        except Exception:
            return []

    def insert_document(self, doc) -> Dict:
        """Legacy interface: Insert a Document object"""
        chunks = []
        for i, chunk in enumerate(doc.chunks):
            chunks.append({
                "id": chunk.get("chunk_id", f"{doc.doc_id}_chunk_{i}"),
                "content": chunk["text"],
                "source": chunk.get("source", doc.filename),
                "chunk_index": i
            })
        count = self.add_documents(chunks)
        return {"chunks_inserted": count}

    def insert_many(self, docs) -> Dict:
        """Insert multiple Document objects"""
        total = 0
        for doc in docs:
            result = self.insert_document(doc)
            total += result["chunks_inserted"]
        return {"total_inserted": total}