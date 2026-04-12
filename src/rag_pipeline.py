"""
RAG Pipeline - Retrieval-Augmented Generation with Groq LLM
"""

import time
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass, field
from datetime import datetime

from groq import Groq

from src.config import Config
from src.vector_store import VectorStore, SearchResult


@dataclass
class RAGResponse:
    """A complete RAG response with sources and metadata"""
    answer: str
    query: str
    sources: List[Dict[str, Any]]
    context_used: str
    model: str
    tokens_used: int
    latency_ms: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 0.0


class RAGPipeline:
    """Production-ready RAG pipeline with Groq"""

    SYSTEM_PROMPT = """You are an expert document analyst and AI assistant.
Your role is to answer questions based ONLY on the provided context documents.

Guidelines:
- Answer clearly and concisely based on the context
- Always cite your sources with [Source: filename] format
- If the context doesn't contain enough information, say so clearly
- Highlight key insights and important details
- Structure longer answers with bullet points or numbered lists
- Never make up information not present in the context"""

    def __init__(self):
        self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
        self.vector_store = VectorStore()
        self.conversation_history: List[Dict[str, str]] = []
        self.query_count = 0
        self.total_tokens = 0

    # ─── Core RAG ─────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int = None,
        filters: Optional[Dict] = None,
        use_history: bool = True
    ) -> RAGResponse:
        """Main RAG query: retrieve context → augment → generate answer"""
        start_time = time.time()
        top_k = top_k or Config.TOP_K_RESULTS

        print(f"\n🔍 Query: {question[:80]}...")

        # Step 1: Retrieve relevant chunks
        results = self.vector_store.semantic_search(
            query=question,
            top_k=top_k,
            filters=filters
        )

        if not results:
            return self._no_context_response(question)

        print(f"   📚 Retrieved {len(results)} chunks (top score: {results[0].score:.3f})")

        # Step 2: Build context
        context = self._build_context(results)

        # Step 3: Generate answer
        messages = self._build_messages(question, context, use_history)
        response_text, tokens = self._call_groq(messages)

        # Step 4: Calculate confidence from retrieval scores
        avg_score = sum(r.score for r in results) / len(results)
        confidence = min(avg_score * 1.2, 1.0)

        # Step 5: Package response
        latency_ms = int((time.time() - start_time) * 1000)
        rag_response = RAGResponse(
            answer=response_text,
            query=question,
            sources=self._format_sources(results),
            context_used=context,
            model=Config.GROQ_MODEL,
            tokens_used=tokens,
            latency_ms=latency_ms,
            confidence=confidence
        )

        # Update state
        self.query_count += 1
        self.total_tokens += tokens
        if use_history:
            self._update_history(question, response_text)

        print(f"   ✅ Answer generated in {latency_ms}ms ({tokens} tokens)")
        return rag_response

    def stream_query(self, question: str, top_k: int = None) -> Generator[str, None, None]:
        """Stream the RAG response token by token"""
        top_k = top_k or Config.TOP_K_RESULTS

        results = self.vector_store.semantic_search(question, top_k=top_k)
        if not results:
            yield "I couldn't find relevant information in the documents."
            return

        context = self._build_context(results)
        messages = self._build_messages(question, context)

        stream = self.groq_client.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=messages,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ─── Context Building ─────────────────────────────────────────────────────

    def _build_context(self, results: List[SearchResult]) -> str:
        """Build a formatted context string from search results"""
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Document {i} | Source: {result.source} | Relevance: {result.score:.2f}]\n"
                f"{result.text}"
            )
        return "\n\n---\n\n".join(context_parts)

    def _build_messages(
        self,
        question: str,
        context: str,
        use_history: bool = True
    ) -> List[Dict[str, str]]:
        """Build the full message list for the LLM"""
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # Include conversation history
        if use_history and self.conversation_history:
            messages.extend(self.conversation_history[-Config.MAX_HISTORY_TURNS * 2:])

        # Add context + question
        user_message = (
            f"Context Documents:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Please answer based on the context above."
        )
        messages.append({"role": "user", "content": user_message})
        return messages

    # ─── LLM Calls ───────────────────────────────────────────────────────────

    def _call_groq(self, messages: List[Dict[str, str]]) -> tuple[str, int]:
        """Call Groq API and return (response_text, tokens_used)"""
        response = self.groq_client.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=messages,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        text = response.choices[0].message.content
        tokens = response.usage.total_tokens
        return text, tokens

    # ─── Advanced Queries ─────────────────────────────────────────────────────

    def multi_hop_query(self, question: str) -> RAGResponse:
        """Multi-hop reasoning: decompose → retrieve per sub-query → synthesize"""
        print(f"\n🔗 Multi-hop query: {question[:60]}...")

        # Step 1: Decompose question
        decompose_prompt = f"""Break this question into 2-3 simpler sub-questions that together would answer it.
Return ONLY the sub-questions, one per line.
Question: {question}"""

        decompose_msg = [
            {"role": "system", "content": "You are a question decomposition expert."},
            {"role": "user", "content": decompose_prompt}
        ]
        sub_questions_text, _ = self._call_groq(decompose_msg)
        sub_questions = [q.strip() for q in sub_questions_text.strip().split("\n") if q.strip()]
        print(f"   📋 Sub-questions: {sub_questions}")

        # Step 2: Retrieve for each sub-question
        all_results = self.vector_store.multi_query_search(sub_questions, top_k=3)

        # Step 3: Synthesize
        context = self._build_context(all_results)
        synthesis_prompt = (
            f"Original Question: {question}\n\n"
            f"Sub-questions explored: {', '.join(sub_questions)}\n\n"
            f"Context:\n{context}\n\n"
            f"Provide a comprehensive answer to the original question."
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": synthesis_prompt}
        ]
        answer, tokens = self._call_groq(messages)

        return RAGResponse(
            answer=answer,
            query=question,
            sources=self._format_sources(all_results),
            context_used=context,
            model=Config.GROQ_MODEL,
            tokens_used=tokens,
            latency_ms=0,
            confidence=0.85
        )

    def summarize_document(self, doc_id: str) -> str:
        """Generate a summary of a specific document"""
        try:
            results = self.vector_store.filtered_search(
                query="main topics key points summary",
                top_k=8
            )
            doc_chunks = [r for r in results if r.doc_id == doc_id]

            if not doc_chunks:
                return "Document not found in vector store."

            context = "\n\n".join(c.text for c in doc_chunks)
            messages = [
                {"role": "system", "content": "You are an expert summarizer."},
                {"role": "user", "content": f"Summarize this document comprehensively:\n\n{context}"}
            ]
            summary, _ = self._call_groq(messages)
            return summary
        except Exception as e:
            return f"Error generating summary: {e}"

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _format_sources(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Format sources for the response object"""
        seen = set()
        sources = []
        for r in results:
            if r.source not in seen:
                seen.add(r.source)
                sources.append({
                    "source": r.source,
                    "doc_id": r.doc_id,
                    "relevance_score": round(r.score, 3),
                    "preview": r.text[:150] + "..."
                })
        return sources

    def _no_context_response(self, question: str) -> RAGResponse:
        """Return a response when no context is found"""
        return RAGResponse(
            answer="I couldn't find relevant information in the uploaded documents to answer your question. Please upload relevant documents first.",
            query=question,
            sources=[],
            context_used="",
            model=Config.GROQ_MODEL,
            tokens_used=0,
            latency_ms=0,
            confidence=0.0
        )

    def _update_history(self, question: str, answer: str):
        """Add to conversation history"""
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})
        # Keep history bounded
        if len(self.conversation_history) > Config.MAX_HISTORY_TURNS * 2:
            self.conversation_history = self.conversation_history[-Config.MAX_HISTORY_TURNS * 2:]

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🗑️ Conversation history cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Return pipeline statistics"""
        return {
            "total_queries": self.query_count,
            "total_tokens_used": self.total_tokens,
            "avg_tokens_per_query": self.total_tokens // max(self.query_count, 1),
            "history_turns": len(self.conversation_history) // 2,
            "model": Config.GROQ_MODEL
        }