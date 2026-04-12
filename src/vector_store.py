"""
Vector Store - All Endee operations: CRUD, search, filtering, batch ops
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from endee import Endee
except ImportError:
    raise ImportError("Endee not installed. Run: pip install endee")

from src.config import Config
from src.embedding_engine import EmbeddingEngine
from src.document_processor import Document


@dataclass
class SearchResult:
    """A single search result with score and metadata"""
    chunk_id: str
    text: str
    score: float
    source: str
    doc_id: str
    metadata: Dict[str, Any]


class VectorStore:
    """Complete Endee vector database operations"""

    def __init__(self):
        self.client = Endee()
        self.embedding_engine = EmbeddingEngine()
        self.collection_name = Config.COLLECTION_NAME
        self._ensure_collection()

    # ─── Collection Management ────────────────────────────────────────────────

    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.list_collections()
            existing = [c.name for c in collections]

            if self.collection_name not in existing:
                self.client.create_collection(
                    name=self.collection_name,
                    dimension=Config.EMBEDDING_DIMENSION,
                    metric="cosine"
                )
                print(f"✅ Created collection: {self.collection_name}")
            else:
                print(f"📦 Using existing collection: {self.collection_name}")
        except Exception as e:
            print(f"⚠️ Collection setup warning: {e}")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get info about the current collection"""
        try:
            info = self.client.describe_collection(self.collection_name)
            return {
                "name": info.name,
                "vector_count": info.vectors_count,
                "dimension": info.config.params.vectors.size,
                "status": info.status
            }
        except Exception as e:
            return {"error": str(e)}

    # ─── CRUD Operations ──────────────────────────────────────────────────────

    def insert_document(self, document: Document) -> Dict[str, Any]:
        """Insert all chunks from a document into the vector store"""
        print(f"\n📥 Inserting: {document.filename} ({document.chunk_count} chunks)")
        start_time = time.time()

        texts = [chunk["text"] for chunk in document.chunks]
        embeddings = self.embedding_engine.encode_batch(texts)

        points = []
        for chunk, embedding in zip(document.chunks, embeddings):
            points.append({
                "id": chunk["chunk_id"],
                "vector": embedding.tolist(),
                "payload": {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "doc_id": chunk["doc_id"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk["total_chunks"],
                    "filename": document.filename,
                    "file_type": document.file_type,
                    "word_count": len(chunk["text"].split()),
                }
            })

        # Batch insert in groups of 100
        batch_size = 100
        inserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)
            inserted += len(batch)
            print(f"   Inserted batch {i // batch_size + 1}: {inserted}/{len(points)} vectors")

        elapsed = time.time() - start_time
        result = {
            "doc_id": document.doc_id,
            "filename": document.filename,
            "chunks_inserted": len(points),
            "elapsed_seconds": round(elapsed, 2)
        }
        print(f"   ✅ Done in {elapsed:.2f}s")
        return result

    def update_chunk(self, chunk_id: str, new_text: str) -> bool:
        """Update a specific chunk's text and re-embed it"""
        try:
            new_embedding = self.embedding_engine.encode(new_text)
            self.client.upsert(
                collection_name=self.collection_name,
                points=[{
                    "id": chunk_id,
                    "vector": new_embedding.tolist(),
                    "payload": {"text": new_text, "updated": True}
                }]
            )
            return True
        except Exception as e:
            print(f"❌ Update failed: {e}")
            return False

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document"""
        try:
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector={"filter": {"must": [{"key": "doc_id", "match": {"value": doc_id}}]}}
            )
            deleted_count = getattr(result, "deleted", 0)
            print(f"🗑️ Deleted {deleted_count} chunks for doc_id: {doc_id}")
            return deleted_count
        except Exception as e:
            print(f"❌ Delete failed: {e}")
            return 0

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific chunk by ID"""
        try:
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[chunk_id],
                with_payload=True,
                with_vectors=False
            )
            if results:
                point = results[0]
                return {"id": point.id, "payload": point.payload}
            return None
        except Exception as e:
            print(f"❌ Retrieve failed: {e}")
            return None

    # ─── Search Operations ────────────────────────────────────────────────────

    def semantic_search(
        self,
        query: str,
        top_k: int = None,
        score_threshold: float = None,
        filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Perform semantic similarity search"""
        top_k = top_k or Config.TOP_K_RESULTS
        score_threshold = score_threshold or Config.SIMILARITY_THRESHOLD

        query_vector = self.embedding_engine.encode(query)

        search_params = {
            "collection_name": self.collection_name,
            "query_vector": query_vector.tolist(),
            "limit": top_k,
            "score_threshold": score_threshold,
            "with_payload": True
        }

        if filters:
            search_params["query_filter"] = filters

        results = self.client.search(**search_params)

        return [
            SearchResult(
                chunk_id=str(r.id),
                text=r.payload.get("text", ""),
                score=r.score,
                source=r.payload.get("source", "unknown"),
                doc_id=r.payload.get("doc_id", ""),
                metadata=r.payload
            )
            for r in results
        ]

    def filtered_search(
        self,
        query: str,
        file_type: Optional[str] = None,
        source: Optional[str] = None,
        min_word_count: Optional[int] = None,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search with metadata filters"""
        must_conditions = []

        if file_type:
            must_conditions.append({"key": "file_type", "match": {"value": file_type}})
        if source:
            must_conditions.append({"key": "source", "match": {"value": source}})
        if min_word_count:
            must_conditions.append({"key": "word_count", "range": {"gte": min_word_count}})

        filters = {"must": must_conditions} if must_conditions else None
        return self.semantic_search(query, top_k=top_k, filters=filters)

    def multi_query_search(self, queries: List[str], top_k: int = 3) -> List[SearchResult]:
        """Search with multiple queries and merge results"""
        all_results: Dict[str, SearchResult] = {}

        for query in queries:
            results = self.semantic_search(query, top_k=top_k)
            for r in results:
                if r.chunk_id not in all_results or r.score > all_results[r.chunk_id].score:
                    all_results[r.chunk_id] = r

        # Sort by score descending
        merged = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
        return merged[:top_k * 2]

    # ─── Batch Operations ─────────────────────────────────────────────────────

    def insert_many(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Insert multiple documents in sequence"""
        results = []
        print(f"\n📦 Batch inserting {len(documents)} documents...")
        for doc in documents:
            result = self.insert_document(doc)
            results.append(result)
        total_chunks = sum(r["chunks_inserted"] for r in results)
        print(f"\n🎉 Batch complete: {len(documents)} docs, {total_chunks} total chunks")
        return results

    def delete_many(self, doc_ids: List[str]) -> Dict[str, int]:
        """Delete multiple documents"""
        total_deleted = 0
        for doc_id in doc_ids:
            total_deleted += self.delete_document(doc_id)
        return {"deleted_count": total_deleted, "doc_count": len(doc_ids)}

    # ─── Analytics ───────────────────────────────────────────────────────────

    def get_all_sources(self) -> List[str]:
        """List all unique document sources"""
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=["source"],
                limit=1000
            )
            sources = set()
            for point in results[0]:
                source = point.payload.get("source")
                if source:
                    sources.add(source)
            return sorted(list(sources))
        except Exception as e:
            return []

    def get_store_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        info = self.get_collection_info()
        sources = self.get_all_sources()
        return {
            "collection": self.collection_name,
            "total_vectors": info.get("vector_count", 0),
            "unique_sources": len(sources),
            "sources": sources,
            "embedding_model": Config.EMBEDDING_MODEL,
            "dimension": Config.EMBEDDING_DIMENSION
        }