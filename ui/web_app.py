"""
Flask Web UI - Dashboard for EndeeRAG
"""

import os
import json
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename

from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context

from src.config import Config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline
from src.agents import MultiAgentSystem

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB
app.config["UPLOAD_FOLDER"] = "data/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize components
processor = DocumentProcessor()
vector_store = VectorStore()
pipeline = RAGPipeline()
mas = MultiAgentSystem()

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EndeeRAG - Document Intelligence</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f1117; color: #e0e0e0; min-height: 100vh; }
.header { background: linear-gradient(135deg, #1a1f2e, #252b3b);
          padding: 20px 32px; border-bottom: 1px solid #2a3040;
          display: flex; align-items: center; gap: 12px; }
.logo { font-size: 24px; font-weight: 700; color: #6366f1; }
.subtitle { color: #666; font-size: 13px; margin-top: 2px; }
.container { display: grid; grid-template-columns: 340px 1fr;
             gap: 0; height: calc(100vh - 65px); }
.sidebar { background: #141820; border-right: 1px solid #2a3040;
           padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 20px; }
.main { display: flex; flex-direction: column; }
.card { background: #1a1f2e; border: 1px solid #2a3040; border-radius: 10px; padding: 16px; }
.card h3 { font-size: 13px; font-weight: 600; color: #8b9ab0; text-transform: uppercase;
           letter-spacing: 0.05em; margin-bottom: 12px; }
.upload-zone { border: 2px dashed #3a4055; border-radius: 8px; padding: 20px;
               text-align: center; cursor: pointer; transition: all 0.2s; color: #666; }
.upload-zone:hover { border-color: #6366f1; color: #6366f1; }
.btn { padding: 8px 16px; border-radius: 6px; border: none; cursor: pointer;
       font-size: 13px; font-weight: 500; transition: all 0.2s; }
.btn-primary { background: #6366f1; color: white; }
.btn-primary:hover { background: #5254cc; }
.btn-outline { background: transparent; color: #8b9ab0; border: 1px solid #3a4055; }
.btn-outline:hover { border-color: #6366f1; color: #6366f1; }
.btn-sm { padding: 5px 10px; font-size: 12px; }
.source-item { display: flex; align-items: center; justify-content: space-between;
               padding: 8px 10px; background: #0f1117; border-radius: 6px;
               margin-bottom: 6px; font-size: 12px; }
.source-dot { width: 6px; height: 6px; border-radius: 50%; background: #6366f1;
              margin-right: 8px; flex-shrink: 0; }
.chat-area { flex: 1; overflow-y: auto; padding: 24px; display: flex;
             flex-direction: column; gap: 16px; }
.message { max-width: 800px; }
.message.user { align-self: flex-end; }
.message.user .bubble { background: #6366f1; color: white; border-radius: 12px 12px 2px 12px; }
.message.assistant .bubble { background: #1a1f2e; border: 1px solid #2a3040;
                              border-radius: 12px 12px 12px 2px; }
.bubble { padding: 12px 16px; font-size: 14px; line-height: 1.6; }
.meta { font-size: 11px; color: #666; margin-top: 6px; display: flex; gap: 12px; }
.sources-bar { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
.source-badge { background: #252b3b; border: 1px solid #3a4055; border-radius: 4px;
                padding: 2px 8px; font-size: 11px; color: #8b9ab0; }
.input-area { padding: 16px 24px; border-top: 1px solid #2a3040;
              background: #141820; display: flex; gap: 10px; align-items: flex-end; }
.input-area textarea { flex: 1; background: #1a1f2e; border: 1px solid #3a4055;
                       color: #e0e0e0; border-radius: 8px; padding: 10px 14px;
                       font-size: 14px; resize: none; font-family: inherit;
                       line-height: 1.5; min-height: 44px; max-height: 120px; }
.input-area textarea:focus { outline: none; border-color: #6366f1; }
.mode-select { display: flex; gap: 6px; padding: 0 24px 12px; background: #141820; }
.mode-btn { padding: 5px 12px; border-radius: 20px; border: 1px solid #3a4055;
            background: transparent; color: #666; font-size: 12px; cursor: pointer; transition: all 0.2s; }
.mode-btn.active { background: #6366f1; border-color: #6366f1; color: white; }
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.stat-box { background: #0f1117; border-radius: 6px; padding: 10px; text-align: center; }
.stat-num { font-size: 20px; font-weight: 700; color: #6366f1; }
.stat-label { font-size: 11px; color: #666; margin-top: 2px; }
.progress { height: 3px; background: #2a3040; border-radius: 2px; overflow: hidden; }
.progress-bar { height: 100%; background: #6366f1; width: 0; transition: width 0.3s; }
.toast { position: fixed; bottom: 24px; right: 24px; background: #1a1f2e;
         border: 1px solid #2a3040; border-radius: 8px; padding: 12px 16px;
         font-size: 13px; display: none; z-index: 100; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
.toast.show { display: block; }
.toast.success { border-color: #22c55e; color: #22c55e; }
.toast.error { border-color: #ef4444; color: #ef4444; }
#file-input { display: none; }
.thinking { display: flex; gap: 4px; padding: 12px 16px; }
.dot { width: 6px; height: 6px; border-radius: 50%; background: #6366f1;
       animation: bounce 1.2s infinite; }
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce { 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-6px)} }
</style>
</head>
<body>
<div class="header">
  <div>
    <div class="logo">⚡ EndeeRAG</div>
    <div class="subtitle">Advanced Document Intelligence System</div>
  </div>
</div>

<div class="container">
  <!-- Sidebar -->
  <div class="sidebar">

    <!-- Upload -->
    <div class="card">
      <h3>📄 Upload Documents</h3>
      <div class="upload-zone" onclick="document.getElementById('file-input').click()">
        <div style="font-size:28px;margin-bottom:8px">📁</div>
        <div style="font-size:13px">Click to upload</div>
        <div style="font-size:11px;margin-top:4px">PDF, DOCX, TXT, MD</div>
      </div>
      <input type="file" id="file-input" multiple accept=".pdf,.docx,.txt,.md"
             onchange="uploadFiles(this.files)">
      <div class="progress" style="margin-top:10px">
        <div class="progress-bar" id="upload-progress"></div>
      </div>
    </div>

    <!-- Stats -->
    <div class="card">
      <h3>📊 System Stats</h3>
      <div class="stat-grid" id="stats-grid">
        <div class="stat-box"><div class="stat-num" id="stat-vectors">-</div><div class="stat-label">Vectors</div></div>
        <div class="stat-box"><div class="stat-num" id="stat-sources">-</div><div class="stat-label">Documents</div></div>
        <div class="stat-box"><div class="stat-num" id="stat-queries">0</div><div class="stat-label">Queries</div></div>
        <div class="stat-box"><div class="stat-num" id="stat-tokens">0</div><div class="stat-label">Tokens</div></div>
      </div>
      <button class="btn btn-outline btn-sm" style="width:100%;margin-top:10px" onclick="refreshStats()">↻ Refresh</button>
    </div>

    <!-- Sources -->
    <div class="card" style="flex:1">
      <h3>📚 Loaded Documents</h3>
      <div id="sources-list"><div style="color:#666;font-size:12px">No documents loaded</div></div>
    </div>

  </div>

  <!-- Main Chat -->
  <div class="main">
    <div class="mode-select">
      <button class="mode-btn active" onclick="setMode('rag', this)">🔍 RAG</button>
      <button class="mode-btn" onclick="setMode('stream', this)">⚡ Stream</button>
      <button class="mode-btn" onclick="setMode('agents', this)">🤖 Multi-Agent</button>
      <button class="mode-btn" onclick="setMode('multihop', this)">🔗 Multi-Hop</button>
    </div>

    <div class="chat-area" id="chat-area">
      <div style="text-align:center;color:#444;padding:40px">
        <div style="font-size:48px;margin-bottom:16px">🧠</div>
        <div style="font-size:18px;color:#666">Upload documents and start asking questions</div>
        <div style="font-size:13px;color:#444;margin-top:8px">Powered by Endee Vector DB + Groq LLM</div>
      </div>
    </div>

    <div class="input-area">
      <textarea id="query-input" placeholder="Ask a question about your documents..."
                rows="1" onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
      <button class="btn btn-primary" onclick="sendQuery()">Send →</button>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
let currentMode = 'rag';
let queryCount = 0;
let totalTokens = 0;

function setMode(mode, btn) {
  currentMode = mode;
  document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuery(); }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function showToast(msg, type = 'success') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = `toast show ${type}`;
  setTimeout(() => t.classList.remove('show'), 3000);
}

function addMessage(role, content, meta = {}) {
  const area = document.getElementById('chat-area');
  const div = document.createElement('div');
  div.className = `message ${role}`;

  let metaHtml = '';
  if (meta.latency) metaHtml += `<span>⏱ ${meta.latency}ms</span>`;
  if (meta.confidence) metaHtml += `<span>🎯 ${Math.round(meta.confidence * 100)}%</span>`;
  if (meta.tokens) metaHtml += `<span>🔤 ${meta.tokens} tokens</span>`;
  if (meta.mode) metaHtml += `<span>🤖 ${meta.mode}</span>`;

  let sourcesHtml = '';
  if (meta.sources && meta.sources.length > 0) {
    sourcesHtml = '<div class="sources-bar">' +
      meta.sources.map(s => `<span class="source-badge">📄 ${s.source}</span>`).join('') +
      '</div>';
  }

  div.innerHTML = `
    <div class="bubble">${content.replace(/\n/g, '<br>')}</div>
    ${metaHtml ? `<div class="meta">${metaHtml}</div>` : ''}
    ${sourcesHtml}
  `;
  area.appendChild(div);
  area.scrollTop = area.scrollHeight;
  return div;
}

function addThinking() {
  const area = document.getElementById('chat-area');
  const div = document.createElement('div');
  div.className = 'message assistant';
  div.id = 'thinking-bubble';
  div.innerHTML = '<div class="bubble"><div class="thinking"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div></div>';
  area.appendChild(div);
  area.scrollTop = area.scrollHeight;
}

function removeThinking() {
  const t = document.getElementById('thinking-bubble');
  if (t) t.remove();
}

async function sendQuery() {
  const input = document.getElementById('query-input');
  const query = input.value.trim();
  if (!query) return;

  input.value = '';
  input.style.height = 'auto';
  addMessage('user', query);
  addThinking();
  queryCount++;
  document.getElementById('stat-queries').textContent = queryCount;

  try {
    if (currentMode === 'stream') {
      removeThinking();
      await streamQuery(query);
    } else {
      const endpoint = {
        'rag': '/api/query',
        'agents': '/api/agents',
        'multihop': '/api/multihop'
      }[currentMode] || '/api/query';

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question: query})
      });
      const data = await res.json();
      removeThinking();

      if (data.error) {
        addMessage('assistant', `❌ Error: ${data.error}`);
        showToast(data.error, 'error');
      } else {
        addMessage('assistant', data.answer, {
          latency: data.latency_ms,
          confidence: data.confidence,
          tokens: data.tokens_used,
          sources: data.sources,
          mode: currentMode
        });
        if (data.tokens_used) {
          totalTokens += data.tokens_used;
          document.getElementById('stat-tokens').textContent = totalTokens;
        }
      }
    }
  } catch (err) {
    removeThinking();
    addMessage('assistant', `❌ Error: ${err.message}`);
    showToast(err.message, 'error');
  }
}

async function streamQuery(query) {
  const area = document.getElementById('chat-area');
  const div = document.createElement('div');
  div.className = 'message assistant';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  div.appendChild(bubble);
  area.appendChild(div);

  const res = await fetch('/api/stream', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question: query})
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let text = '';

  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    text += decoder.decode(value);
    bubble.innerHTML = text.replace(/\n/g, '<br>');
    area.scrollTop = area.scrollHeight;
  }
}

async function uploadFiles(files) {
  const bar = document.getElementById('upload-progress');
  bar.style.width = '10%';

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const formData = new FormData();
    formData.append('file', file);

    try {
      bar.style.width = `${((i + 1) / files.length) * 90}%`;
      const res = await fetch('/api/upload', {method: 'POST', body: formData});
      const data = await res.json();

      if (data.error) {
        showToast(`Failed: ${file.name}`, 'error');
      } else {
        showToast(`✅ Loaded: ${file.name} (${data.chunks} chunks)`);
      }
    } catch (err) {
      showToast(`Error uploading ${file.name}`, 'error');
    }
  }

  bar.style.width = '100%';
  setTimeout(() => { bar.style.width = '0'; }, 1000);
  refreshStats();
}

async function refreshStats() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    document.getElementById('stat-vectors').textContent = data.total_vectors || 0;
    document.getElementById('stat-sources').textContent = data.unique_sources || 0;

    const list = document.getElementById('sources-list');
    if (data.sources && data.sources.length > 0) {
      list.innerHTML = data.sources.map(s =>
        `<div class="source-item">
          <div style="display:flex;align-items:center">
            <div class="source-dot"></div>
            <span>${s}</span>
          </div>
        </div>`
      ).join('');
    } else {
      list.innerHTML = '<div style="color:#666;font-size:12px">No documents loaded</div>';
    }
  } catch (err) {
    console.error('Stats error:', err);
  }
}

refreshStats();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        doc = processor.process_file(save_path)
        result = vector_store.insert_document(doc)
        return jsonify({"success": True, "filename": filename, "chunks": result["chunks_inserted"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        response = pipeline.query(question)
        return jsonify({
            "answer": response.answer,
            "sources": response.sources,
            "latency_ms": response.latency_ms,
            "confidence": response.confidence,
            "tokens_used": response.tokens_used
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stream", methods=["POST"])
def stream():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return Response("No question provided", status=400)

    def generate():
        for token in pipeline.stream_query(question):
            yield token

    return Response(stream_with_context(generate()), mimetype="text/plain")


@app.route("/api/agents", methods=["POST"])
def agents():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        result = mas.run(question, verbose=False)
        return jsonify({
            "answer": result.final_answer,
            "sources": result.sources,
            "agents_used": result.agents_used,
            "latency_ms": result.latency_ms,
            "tokens_used": result.total_tokens,
            "confidence": 0.9
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/multihop", methods=["POST"])
def multihop():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        response = pipeline.multi_hop_query(question)
        return jsonify({
            "answer": response.answer,
            "sources": response.sources,
            "latency_ms": response.latency_ms,
            "tokens_used": response.tokens_used,
            "confidence": response.confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def stats():
    try:
        vs_stats = vector_store.get_store_stats()
        rag_stats = pipeline.get_stats()
        return jsonify({**vs_stats, **rag_stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n🚀 Starting EndeeRAG Web UI...")
    print("   Open: http://localhost:5000\n")
    app.run(debug=True, port=5000)