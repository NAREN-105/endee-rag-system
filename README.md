# EndeeRAG — Advanced Document Intelligence System

A production-ready RAG system built with **Endee Vector Database** and **Groq LLM**.

## Features
- Multi-format ingestion: PDF, DOCX, TXT, MD
- Semantic search with Endee vector DB
- Multi-agent pipeline (Router → Researcher → Analyst → Synthesizer)
- Streaming responses
- Multi-hop reasoning
- Flask web dashboard
- CLI interface

## Quick Start

```bash
# 1. Add API keys to .env
GROQ_API_KEY=gsk_...
ENDEE_API_KEY=...

# 2. Demo (no API keys needed)
python main.py

# 3. Web UI
python ui/web_app.py
# Open http://localhost:5000

# 4. CLI
python main.py --mode cli

# 5. Quick test
python main.py --test
```

## Architecture

```
Documents → Processor → Embeddings → Endee Vector DB
                                           ↓
User Query → Embedding → Semantic Search → RAG Pipeline → Groq LLM → Answer
                                           ↓
                                    Multi-Agent System
                           Router → Researcher → Analyst → Synthesizer
```

## Endee Operations Used
- `create_collection` — Initialize vector store
- `upsert` — Insert/update vectors with metadata
- `search` — Semantic similarity search
- `delete` — Remove documents by filter
- `scroll` — List all stored vectors
- `retrieve` — Fetch specific vectors by ID
- `describe_collection` — Collection statistics

## Tech Stack
| Component | Technology |
|-----------|-----------|
| Vector DB | Endee |
| LLM | Groq (llama-3.3-70b) |
| Embeddings | sentence-transformers |
| Web UI | Flask |
| Doc Processing | PyPDF2, python-docx |