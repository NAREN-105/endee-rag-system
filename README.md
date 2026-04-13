# 🚀 EndeeRAG — Advanced Document Intelligence System

![Endee](https://img.shields.io/badge/Endee-Vector%20DB-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3--70b-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-orange?style=for-the-badge)

> A production-ready RAG system built with **Endee Vector Database** and **Groq LLM** featuring a 4-agent pipeline for advanced document intelligence.

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

Professionals spend 35–40% of their time searching through large documents manually. Traditional keyword search fails to understand meaning and context.

**EndeeRAG solves this with:**
- ✅ Semantic search using vector embeddings
- ✅ Multi-agent AI pipeline for deep reasoning
- ✅ Support for PDF, DOCX, TXT, MD formats
- ✅ Streaming responses via Groq LLM

---

## ✨ Features

- 📄 **Multi-format ingestion** — PDF, DOCX, TXT, MD
- 🔍 **Semantic search** — powered by Endee Vector DB
- 🤖 **Multi-agent pipeline** — Router → Researcher → Analyst → Synthesizer
- ⚡ **Streaming responses** — real-time output
- 🧠 **Multi-hop reasoning** — across multiple documents
- 🌐 **Flask web dashboard** — at localhost:5000
- 💻 **CLI interface** — for power users

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                       │
│                                                             │
│  📄 Documents → 🔧 Processor → 🔢 Embeddings → 🗄️ Endee DB  │
│  (PDF/DOCX/      (Chunking &    (sentence-      (Vector      │
│   TXT/MD)         Cleaning)      transformers)   Storage)    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                          │
│                                                             │
│  ❓ User Query → 🔢 Embedding → 🔍 Semantic Search → 📦 Top-K │
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

## 🔷 Endee Operations Used

| Operation | Purpose |
|-----------|---------|
| `create_collection` | Initialize vector store |
| `upsert` | Insert/update vectors with metadata |
| `search` | Semantic similarity search |
| `delete` | Remove documents by filter |
| `scroll` | List all stored vectors |
| `retrieve` | Fetch specific vectors by ID |
| `describe_collection` | Collection statistics |

> ✅ Uses **all 7 Endee operations** — most comprehensive usage!

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Vector DB | Endee |
| LLM | Groq (llama-3.3-70b) |
| Embeddings | sentence-transformers |
| Web UI | Flask |
| Doc Processing | PyPDF2, python-docx |

---

## 🤖 Multi-Agent Pipeline

| Agent | Role |
|-------|------|
| **Router** | Classifies query type |
| **Researcher** | Retrieves relevant context from Endee |
| **Analyst** | Evaluates evidence quality |
| **Synthesizer** | Drafts the final answer via Groq LLM |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/NAREN-105/endee-rag-system.git
cd endee-rag-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add API keys to .env
GROQ_API_KEY=gsk_...

# 4. Run demo (no API keys needed)
python main.py

# 5. Run Web UI → http://localhost:5000
python ui/web_app.py

# 6. Run CLI mode
python main.py --mode cli
```

---

## ✅ Internship Requirements Checklist

- ⭐ Starred the official endee-io/endee repository
- 🍴 Forked the official endee-io/endee repository
- ✅ Used Endee as core vector DB (7 operations)
- ✅ Built real-world AI application with RAG pipeline
- ✅ Multi-agent system for document reasoning
- ✅ Streaming responses via Groq LLM
- ✅ Public GitHub repository

---

## 📄 License

MIT License

---

**Built with ❤️ by Narendaran M | V.S.B Engineering College | Endee Internship 2026**