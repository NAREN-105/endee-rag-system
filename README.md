# 🚀 EndeeRAG — Advanced Document Intelligence System

![Endee](https://img.shields.io/badge/Endee-Vector%20DB-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3--70b-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-orange?style=for-the-badge)

> A production-ready RAG (Retrieval-Augmented Generation) system built with **Endee Vector Database** and **Groq LLM**, featuring a 4-agent AI pipeline, multi-hop reasoning, streaming responses, and a real-time Flask web dashboard.

---

## 👤 Candidate Information

| Field | Details |
|-------|---------|
| **Name** | Narendaran M |
| **GitHub** | [@NAREN-105](https://github.com/NAREN-105) |
| **Institution** | V.S.B Engineering College |
| **Project** | EndeeRAG — Advanced Document Intelligence System |
| **Submission Date** | April 2026 |

---

## 🎯 Problem Statement

Professionals spend **35–40% of their time** searching through large documents manually. Traditional keyword search fails to understand meaning and context, leading to missed information and wasted hours.

**EndeeRAG solves this by providing:**

- ✅ **Semantic search** using vector embeddings — finds information by *meaning*, not just keywords
- ✅ **Multi-agent AI pipeline** — 4 specialized agents collaborate to provide deep, accurate answers
- ✅ **Multi-hop reasoning** — breaks complex questions into sub-questions for comprehensive answers
- ✅ **Real-time streaming** — watch AI generate answers word-by-word
- ✅ **Multi-format support** — PDF, DOCX, TXT, and Markdown files

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **Multi-Format Ingestion** | Upload and process PDF, DOCX, TXT, and MD files |
| 🔍 **Semantic Search** | Vector similarity search powered by Endee DB |
| 🤖 **4-Agent AI Pipeline** | Router → Researcher → Analyst → Synthesizer |
| ⚡ **Streaming Responses** | Real-time token-by-token output via Groq |
| 🔗 **Multi-Hop Reasoning** | Decomposes complex questions into sub-queries |
| 🌐 **Flask Web Dashboard** | Beautiful dark-mode UI at `localhost:5000` |
| 💻 **CLI Interface** | Full command-line mode for power users |
| 📊 **Live System Stats** | Real-time vector count, query count, and token usage |
| 🧠 **Conversation Memory** | Maintains context across follow-up questions |
| 📦 **Smart Chunking** | Sentence-aware text splitting with configurable overlap |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                       │
│                                                             │
│  📄 Documents → 🔧 Processor → 🔢 Embeddings → 🗄️ Endee DB  │
│  (PDF/DOCX/      (Smart         (sentence-      (Vector     │
│   TXT/MD)         Chunking)      transformers)   Storage)    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                          │
│                                                             │
│  ❓ User Query → 🔢 Embedding → 🔍 Semantic Search → 📦 Top-K│
│                                      (Endee)        Chunks  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   MULTI-AGENT PIPELINE                       │
│                                                             │
│   🔀 Router → 🔬 Researcher → 🧠 Analyst → ✍️ Synthesizer   │
│   (Classify)   (Fetch Context) (Evaluate)   (Draft Answer)  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                           │
│                                                             │
│         ⚡ Groq LLM (llama-3.3-70b) → 💬 Answer            │
│              (Streaming Response)                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔷 Endee Database Operations Used

This project demonstrates **comprehensive usage of the Endee Vector Database SDK**, utilizing all major operations:

| # | Operation | Method Used | Purpose |
|---|-----------|-------------|---------|
| 1 | **Create Index** | `client.create_index()` | Initialize the `endee_rag` vector index with 384 dimensions and cosine distance |
| 2 | **List Indexes** | `client.list_indexes()` | Check if the index already exists before creating |
| 3 | **Get Index** | `client.get_index()` | Retrieve the index object for upsert/query operations |
| 4 | **Upsert Vectors** | `index.upsert()` | Insert document chunk embeddings with metadata (source, content, chunk_index) |
| 5 | **Query / Search** | `index.query()` | Perform semantic similarity search using query embeddings |
| 6 | **Delete** | `index.delete()` | Remove vectors by source filter |
| 7 | **Describe / Stats** | `client.get_index()` | Retrieve collection statistics (vector count, dimensions) |

> ✅ **Uses all 7 Endee operations** — the most comprehensive Endee SDK usage!

### How Endee is Integrated

```python
from endee import Endee

# 1. Connect to Endee (running on localhost:8080)
client = Endee()

# 2. Create an index for storing document vectors
client.create_index(name="endee_rag", dimension=384, space_type="cosine")

# 3. Get the index object
index = client.get_index(name="endee_rag")

# 4. Upsert document vectors with metadata
index.upsert([{
    "id": "doc1_chunk_0",
    "vector": [0.12, -0.34, ...],  # 384-dim embedding
    "meta": {"content": "...", "source": "report.pdf", "chunk_index": 0}
}])

# 5. Query — semantic search
results = index.query(vector=[0.15, -0.28, ...], top_k=5)
for r in results:
    print(f"ID: {r['id']}, Similarity: {r['similarity']:.3f}")
```

---

## 🤖 Multi-Agent Pipeline

The system uses **4 specialized AI agents** that work in sequence to produce high-quality answers:

| Agent | Role | What It Does |
|-------|------|-------------|
| 🔀 **Router** | Classifier | Analyzes the query type (factual, analytical, summary) and decides which agents to invoke |
| 🔬 **Researcher** | Information Retrieval | Searches Endee DB for relevant document chunks and organizes findings |
| 🧠 **Analyst** | Deep Analysis | Identifies patterns, draws conclusions, and evaluates evidence quality |
| ✍️ **Synthesizer** | Answer Generation | Combines all agent outputs into a coherent, cited final answer via Groq LLM |

**Additional agents available:**
- 📝 **Summarizer** — Creates concise summaries of retrieved information
- 🔍 **Critic** — Reviews and validates the analysis quality with a quality score

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vector Database** | [Endee](https://endee.io) | Store and search document embeddings |
| **LLM** | [Groq](https://groq.com) (llama-3.3-70b-versatile) | Generate AI answers |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Convert text to 384-dim vectors |
| **Web Framework** | Flask | Web dashboard and REST API |
| **Document Parsing** | PyPDF2, python-docx | Extract text from PDF/DOCX files |
| **Text Processing** | Custom SmartTextSplitter | Sentence-aware chunking with overlap |
| **Language** | Python 3.10+ | Core application logic |

---

## 📁 Project Structure

```
endee-rag-system/
├── .env                          # API keys (GROQ_API_KEY, ENDEE_API_KEY)
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── main.py                       # CLI entry point (demo/cli/test modes)
│
├── src/                          # Core application modules
│   ├── __init__.py
│   ├── config.py                 # Configuration and environment variables
│   ├── document_processor.py     # Multi-format document ingestion
│   ├── embedding_engine.py       # Embedding generation with caching
│   ├── vector_store.py           # Endee DB operations (upsert, query, etc.)
│   ├── rag_pipeline.py           # RAG query pipeline with Groq LLM
│   └── agents.py                 # Multi-agent system (Router, Researcher, etc.)
│
├── ui/                           # Web interface
│   └── web_app.py                # Flask dashboard with real-time chat UI
│
├── utils/                        # Utility modules
│   ├── __init__.py
│   └── text_splitter.py          # Smart text chunking with overlap
│
├── data/                         # Data storage
│   ├── uploads/                  # Uploaded documents
│   ├── documents/                # Processed documents
│   └── database/                 # Local database files
│
└── tests/                        # Test suite
    └── test_system.py            # System integration tests
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Docker Desktop** (for running Endee server)
- **Groq API Key** — Get one free at [console.groq.com](https://console.groq.com)

### Step 1: Start Endee Database

```bash
docker run -p 8080:8080 -v ./endee-data:/data --name endee-server endeeio/endee-server:latest
```

Verify Endee is running by opening **http://localhost:8080** — you should see the Endee dashboard.

### Step 2: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/NAREN-105/endee-rag-system.git
cd endee-rag-system

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure API Keys

Edit the `.env` file:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
ENDEE_API_KEY=local
ENDEE_URL=http://localhost:8080
```

### Step 4: Run the Application

**Option A — Web Dashboard (Recommended):**

```bash
python ui/web_app.py
# Open http://localhost:5000 in your browser
```

**Option B — CLI Mode:**

```bash
python main.py --mode cli
```

**Option C — Demo Mode (No API keys needed):**

```bash
python main.py --mode demo
```

**Option D — Quick Test:**

```bash
python main.py --mode test
```

---

## 🌐 Web Dashboard

The web dashboard at `http://localhost:5000` provides:

### Left Sidebar
- **📄 Upload Documents** — Drag-and-drop or click to upload PDF, DOCX, TXT, MD files
- **📊 System Stats** — Live vector count, document count, query count, token usage
- **📚 Loaded Documents** — List of all uploaded documents
- **💡 Query Modes** — Explanation of each query mode

### Main Chat Area
- **4 Query Modes** with real-time descriptions:

| Mode | Description |
|------|-------------|
| 🔍 **RAG** | Basic semantic search + AI answer generation |
| ⚡ **Stream** | Same as RAG but streams the response word-by-word |
| 🤖 **Multi-Agent** | 4 AI agents collaborate for thorough analysis |
| 🔗 **Multi-Hop** | Decomposes complex questions into sub-queries |

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web dashboard |
| `/api/upload` | POST | Upload and process a document |
| `/api/query` | POST | RAG query |
| `/api/stream` | POST | Streaming RAG query |
| `/api/agents` | POST | Multi-agent query |
| `/api/multihop` | POST | Multi-hop reasoning query |
| `/api/stats` | GET | System statistics |

---

## 💻 CLI Commands

```
>>> help           # Show available commands
>>> load <path>    # Load a document or directory
>>> ask <question> # Ask a question about loaded documents
>>> stats          # Show system statistics
>>> sources        # List all document sources
>>> clear          # Clear conversation history
>>> exit           # Quit the CLI
```

---

## 🧪 Testing

```bash
# Run quick system test
python main.py --mode test

# Run test suite
python -m pytest tests/
```

---

## 📊 How It Works — Step by Step

### 1. Document Ingestion
```
Upload file → Detect format (PDF/DOCX/TXT/MD) → Extract text
→ Smart chunking (500 chars, 50 char overlap) → Generate embeddings
→ Store vectors + metadata in Endee DB
```

### 2. Query Processing (RAG Mode)
```
User question → Generate query embedding → Endee semantic search (top 5)
→ Build context from matched chunks → Send to Groq LLM with system prompt
→ Return answer with source citations and confidence score
```

### 3. Multi-Agent Mode
```
User question → Router (classify query type) → Researcher (search Endee DB)
→ Analyst (evaluate evidence) → Synthesizer (draft final answer via Groq)
→ Return comprehensive answer with agent trace
```

### 4. Multi-Hop Mode
```
User question → Groq decomposes into 2-3 sub-questions
→ Search Endee DB for each sub-question separately
→ Merge all results → Synthesize comprehensive answer
```

---

## ⚙️ Configuration

All configuration is in `src/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |
| `EMBEDDING_DIMENSION` | 384 | Vector dimensions |
| `TOP_K_RESULTS` | 5 | Number of search results |
| `SIMILARITY_THRESHOLD` | 0.3 | Minimum relevance score |
| `GROQ_MODEL` | llama-3.3-70b-versatile | Groq LLM model |
| `MAX_TOKENS` | 1024 | Max response tokens |
| `TEMPERATURE` | 0.1 | LLM temperature |
| `COLLECTION_NAME` | endee_rag | Endee index name |
| `MAX_HISTORY_TURNS` | 5 | Conversation memory length |

---

## ✅ Internship Requirements Checklist

- [x] ⭐ Starred the official [endee-io/endee](https://github.com/endee-io/endee) repository
- [x] 🍴 Forked the official [endee-io/endee](https://github.com/endee-io/endee) repository
- [x] ✅ Used Endee as core vector database (**7 operations**: create_index, list_indexes, get_index, upsert, query, delete, describe)
- [x] ✅ Built a real-world AI application (RAG pipeline for document intelligence)
- [x] ✅ Multi-agent system with 4+ specialized agents
- [x] ✅ Streaming responses via Groq LLM
- [x] ✅ Multi-hop reasoning for complex queries
- [x] ✅ Web dashboard with real-time UI
- [x] ✅ CLI interface for power users
- [x] ✅ Public GitHub repository with comprehensive documentation

---

## 🔮 Future Improvements

- [ ] Add support for image and table extraction from PDFs
- [ ] Implement user authentication for the web dashboard
- [ ] Add document comparison and diff analysis
- [ ] Support for more LLM providers (OpenAI, Anthropic)
- [ ] Implement vector index backup and restore via Endee
- [ ] Add batch document processing with progress tracking
- [ ] Deploy to cloud with Docker Compose

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ by Narendaran M | V.S.B Engineering College | Endee Internship 2026**

[⭐ Star this repo](https://github.com/NAREN-105/endee-rag-system) · [🐛 Report Bug](https://github.com/NAREN-105/endee-rag-system/issues) · [💡 Request Feature](https://github.com/NAREN-105/endee-rag-system/issues)

</div>