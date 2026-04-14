"""
Flask Web UI - Dashboard for EndeeRAG
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename

# Ensure the project root is on sys.path so src/utils modules can be found
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context

from src.config import Config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline
from src.agents import MultiAgentSystem


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB
app.config["UPLOAD_FOLDER"] = os.path.join(PROJECT_ROOT, "data", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Change CWD to project root so relative paths work
os.chdir(PROJECT_ROOT)

# Initialize components
processor = DocumentProcessor()
vector_store = VectorStore()
pipeline = RAGPipeline()
mas = MultiAgentSystem()

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EndeeRAG - Document Intelligence</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; overflow: hidden; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f1117; color: #e0e0e0; }

/* Header */
.header { background: linear-gradient(135deg, #1a1f2e, #252b3b);
          padding: 16px 32px; border-bottom: 1px solid #2a3040;
          display: flex; align-items: center; justify-content: space-between;
          flex-shrink: 0; height: 60px; }
.logo { font-size: 22px; font-weight: 700; color: #6366f1; }
.subtitle { color: #666; font-size: 12px; margin-top: 2px; }
.header-status { display: flex; align-items: center; gap: 8px; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; }
.status-text { font-size: 11px; color: #666; }

/* Layout */
.app-layout { display: flex; height: calc(100vh - 60px); overflow: hidden; }

/* ======= SIDEBAR with its own scrollbar ======= */
.sidebar { width: 340px; min-width: 340px; background: #141820; border-right: 1px solid #2a3040;
           display: flex; flex-direction: column; overflow: hidden; }
.sidebar-scroll { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 16px; }
.sidebar-scroll::-webkit-scrollbar { width: 6px; }
.sidebar-scroll::-webkit-scrollbar-track { background: transparent; }
.sidebar-scroll::-webkit-scrollbar-thumb { background: #3a4055; border-radius: 3px; }
.sidebar-scroll::-webkit-scrollbar-thumb:hover { background: #6366f1; }

/* ======= MAIN CHAT with its own scrollbar ======= */
.main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.chat-area { flex: 1; overflow-y: auto; padding: 24px; display: flex;
             flex-direction: column; gap: 16px; }
.chat-area::-webkit-scrollbar { width: 6px; }
.chat-area::-webkit-scrollbar-track { background: transparent; }
.chat-area::-webkit-scrollbar-thumb { background: #3a4055; border-radius: 3px; }
.chat-area::-webkit-scrollbar-thumb:hover { background: #6366f1; }

/* Cards */
.card { background: #1a1f2e; border: 1px solid #2a3040; border-radius: 10px; padding: 16px; }
.card h3 { font-size: 12px; font-weight: 600; color: #8b9ab0; text-transform: uppercase;
           letter-spacing: 0.05em; margin-bottom: 12px; }

/* Upload */
.upload-zone { border: 2px dashed #3a4055; border-radius: 8px; padding: 18px;
               text-align: center; cursor: pointer; transition: all 0.2s; color: #666; }
.upload-zone:hover { border-color: #6366f1; color: #6366f1; background: rgba(99,102,241,0.05); }

/* Buttons */
.btn { padding: 8px 16px; border-radius: 6px; border: none; cursor: pointer;
       font-size: 13px; font-weight: 500; transition: all 0.2s; }
.btn-primary { background: #6366f1; color: white; }
.btn-primary:hover { background: #5254cc; }
.btn-outline { background: transparent; color: #8b9ab0; border: 1px solid #3a4055; }
.btn-outline:hover { border-color: #6366f1; color: #6366f1; }
.btn-sm { padding: 5px 10px; font-size: 12px; }
.btn:disabled, .mode-btn:disabled, textarea:disabled { opacity: 0.5; cursor: not-allowed; }

/* Sources list */
.source-item { display: flex; align-items: center; padding: 8px 10px;
               background: #0f1117; border-radius: 6px; margin-bottom: 6px; font-size: 12px; }
.source-dot { width: 6px; height: 6px; border-radius: 50%; background: #6366f1;
              margin-right: 8px; flex-shrink: 0; }

/* Messages */
.message { max-width: 800px; animation: fadeIn 0.3s ease; }
.message.user { align-self: flex-end; }
.message.user .bubble { background: #6366f1; color: white; border-radius: 12px 12px 2px 12px; }
.message.assistant .bubble { background: #1a1f2e; border: 1px solid #2a3040;
                              border-radius: 12px 12px 12px 2px; }
.bubble { padding: 12px 16px; font-size: 14px; line-height: 1.6; }
.meta { font-size: 11px; color: #666; margin-top: 6px; display: flex; gap: 12px; flex-wrap: wrap; }
.sources-bar { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
.source-badge { background: #252b3b; border: 1px solid #3a4055; border-radius: 4px;
                padding: 2px 8px; font-size: 11px; color: #8b9ab0; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

/* Mode select bar */
.mode-bar { padding: 10px 24px; background: #141820; border-bottom: 1px solid #2a3040;
            display: flex; flex-direction: column; gap: 8px; flex-shrink: 0; }
.mode-buttons { display: flex; gap: 6px; }
.mode-btn { padding: 6px 14px; border-radius: 20px; border: 1px solid #3a4055;
            background: transparent; color: #666; font-size: 12px; cursor: pointer; 
            transition: all 0.2s; position: relative; }
.mode-btn:hover { background: #2a3040; border-color: #6366f1; color: #fff; }
.mode-btn.active { background: #6366f1; border-color: #6366f1; color: white; box-shadow: 0 2px 8px rgba(99,102,241,0.3); }
.mode-desc { font-size: 11px; color: #555; padding: 0 2px; line-height: 1.4; }
.mode-desc strong { color: #8b9ab0; }

/* Input area */
.input-area { padding: 14px 24px; border-top: 1px solid #2a3040;
              background: #141820; display: flex; gap: 10px; align-items: flex-end; flex-shrink: 0; }
.input-area textarea { flex: 1; background: #1a1f2e; border: 1px solid #3a4055;
                       color: #e0e0e0; border-radius: 8px; padding: 10px 14px;
                       font-size: 14px; resize: none; font-family: inherit;
                       line-height: 1.5; min-height: 44px; max-height: 120px; }
.input-area textarea:focus { outline: none; border-color: #6366f1; box-shadow: 0 0 0 2px rgba(99,102,241,0.15); }

/* Stats */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.stat-box { background: #0f1117; border-radius: 6px; padding: 10px; text-align: center; }
.stat-num { font-size: 20px; font-weight: 700; color: #6366f1; }
.stat-label { font-size: 11px; color: #666; margin-top: 2px; }
.progress { height: 3px; background: #2a3040; border-radius: 2px; overflow: hidden; }
.progress-bar { height: 100%; background: #6366f1; width: 0; transition: width 0.3s; }

/* Toast */
.toast { position: fixed; bottom: 24px; right: 24px; background: #1a1f2e;
         border: 1px solid #2a3040; border-radius: 8px; padding: 12px 16px;
         font-size: 13px; display: none; z-index: 100; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
.toast.show { display: block; animation: slideUp 0.3s ease; }
.toast.success { border-color: #22c55e; color: #22c55e; }
.toast.error { border-color: #ef4444; color: #ef4444; }
@keyframes slideUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }

#file-input { display: none; }
.thinking { display: flex; gap: 4px; padding: 12px 16px; }
.dot { width: 6px; height: 6px; border-radius: 50%; background: #6366f1;
       animation: bounce 1.2s infinite; }
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce { 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-6px)} }

/* Welcome card */
.welcome { text-align: center; color: #444; padding: 40px 20px; }
.welcome-icon { font-size: 48px; margin-bottom: 16px; }
.welcome-title { font-size: 18px; color: #666; margin-bottom: 8px; }
.welcome-sub { font-size: 13px; color: #444; }

/* Mode info cards in welcome */
.mode-info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; max-width: 600px; margin: 24px auto 0; }
.mode-info-card { background: #1a1f2e; border: 1px solid #2a3040; border-radius: 8px; padding: 14px;
                  text-align: left; }
.mode-info-card .mi-icon { font-size: 20px; margin-bottom: 6px; }
.mode-info-card .mi-title { font-size: 13px; font-weight: 600; color: #e0e0e0; margin-bottom: 4px; }
.mode-info-card .mi-desc { font-size: 11px; color: #666; line-height: 1.4; }
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div>
    <div class="logo">⚡ EndeeRAG</div>
    <div class="subtitle">Advanced Document Intelligence System</div>
  </div>
  <div class="header-status">
    <div class="status-dot"></div>
    <span class="status-text">Endee DB Connected</span>
  </div>
</div>

<!-- App Layout -->
<div class="app-layout">

  <!-- ===== SIDEBAR (own scroll) ===== -->
  <div class="sidebar">
    <div class="sidebar-scroll">

      <!-- Upload Card -->
      <div class="card">
        <h3>📄 Upload Documents</h3>
        <div class="upload-zone" onclick="document.getElementById('file-input').click()">
          <div style="font-size:28px;margin-bottom:6px">📁</div>
          <div style="font-size:13px">Click to upload</div>
          <div style="font-size:11px;margin-top:4px;color:#555">PDF, DOCX, TXT, MD</div>
        </div>
        <input type="file" id="file-input" multiple accept=".pdf,.docx,.txt,.md"
               onchange="uploadFiles(this.files)">
        <div class="progress" style="margin-top:10px">
          <div class="progress-bar" id="upload-progress"></div>
        </div>
      </div>

      <!-- Stats Card -->
      <div class="card">
        <h3>📊 System Stats</h3>
        <div class="stat-grid">
          <div class="stat-box"><div class="stat-num" id="stat-vectors">-</div><div class="stat-label">Vectors</div></div>
          <div class="stat-box"><div class="stat-num" id="stat-sources">-</div><div class="stat-label">Documents</div></div>
          <div class="stat-box"><div class="stat-num" id="stat-queries">0</div><div class="stat-label">Queries</div></div>
          <div class="stat-box"><div class="stat-num" id="stat-tokens">0</div><div class="stat-label">Tokens</div></div>
        </div>
        <button class="btn btn-outline btn-sm" style="width:100%;margin-top:10px" onclick="refreshStats()">↻ Refresh Stats</button>
      </div>

      <!-- Sources Card -->
      <div class="card">
        <h3>📚 Loaded Documents</h3>
        <div id="sources-list"><div style="color:#666;font-size:12px">No documents loaded</div></div>
      </div>

      <!-- Mode Help Card -->
      <div class="card">
        <h3>💡 Query Modes</h3>
        <div style="font-size:12px; color:#8b9ab0; line-height:1.6;">
          <div style="margin-bottom:8px"><strong>🔍 RAG</strong> — Basic search + AI answer. Retrieves relevant chunks from your documents and generates an answer using Groq LLM.</div>
          <div style="margin-bottom:8px"><strong>⚡ Stream</strong> — Same as RAG but streams the answer word-by-word in real-time for a faster feel.</div>
          <div style="margin-bottom:8px"><strong>🤖 Multi-Agent</strong> — Uses 4 AI agents: Router → Researcher → Analyst → Synthesizer for deeper, more thorough analysis.</div>
          <div><strong>🔗 Multi-Hop</strong> — Breaks complex questions into sub-questions, retrieves info for each, then combines everything into one comprehensive answer.</div>
        </div>
      </div>

    </div>
  </div>

  <!-- ===== MAIN AREA (chat has own scroll) ===== -->
  <div class="main">

    <!-- Mode buttons + description -->
    <div class="mode-bar">
      <div class="mode-buttons">
        <button class="mode-btn active" data-mode="rag" onclick="setMode('rag', this)">🔍 RAG</button>
        <button class="mode-btn" data-mode="stream" onclick="setMode('stream', this)">⚡ Stream</button>
        <button class="mode-btn" data-mode="agents" onclick="setMode('agents', this)">🤖 Multi-Agent</button>
        <button class="mode-btn" data-mode="multihop" onclick="setMode('multihop', this)">🔗 Multi-Hop</button>
      </div>
      <div class="mode-desc" id="mode-desc"><strong>RAG Mode:</strong> Search documents → Retrieve context → Generate answer with Groq LLM</div>
    </div>

    <!-- Chat area (own scroll) -->
    <div class="chat-area" id="chat-area">
      <div class="welcome">
        <div class="welcome-icon">🧠</div>
        <div class="welcome-title">Upload documents and start asking questions</div>
        <div class="welcome-sub">Powered by Endee Vector DB + Groq LLM</div>
        <div class="mode-info-grid">
          <div class="mode-info-card"><div class="mi-icon">🔍</div><div class="mi-title">RAG Mode</div><div class="mi-desc">Simple semantic search + AI answer. Best for quick factual questions.</div></div>
          <div class="mode-info-card"><div class="mi-icon">⚡</div><div class="mi-title">Stream Mode</div><div class="mi-desc">Real-time streaming response. Watch the AI think word by word.</div></div>
          <div class="mode-info-card"><div class="mi-icon">🤖</div><div class="mi-title">Multi-Agent</div><div class="mi-desc">4 specialized AI agents collaborate for deep analysis.</div></div>
          <div class="mode-info-card"><div class="mi-icon">🔗</div><div class="mi-title">Multi-Hop</div><div class="mi-desc">Breaks complex questions into sub-questions for comprehensive answers.</div></div>
        </div>
      </div>
    </div>

    <!-- Input -->
    <div class="input-area">
      <textarea id="query-input" placeholder="Ask a question about your documents..."
                rows="1" onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
      <button class="btn btn-primary" id="send-btn" onclick="sendQuery()">Send →</button>
    </div>
  </div>

</div>

<div class="toast" id="toast"></div>

<script>
let currentMode = 'rag';
let queryCount = 0;
let totalTokens = 0;

const MODE_DESCRIPTIONS = {
  'rag': '<strong>RAG Mode:</strong> Search documents → Retrieve context → Generate answer with Groq LLM',
  'stream': '<strong>Stream Mode:</strong> Same as RAG but streams the answer word-by-word in real-time',
  'agents': '<strong>Multi-Agent Mode:</strong> Router → Researcher → Analyst → Synthesizer (4 agents collaborate)',
  'multihop': '<strong>Multi-Hop Mode:</strong> Breaks question into sub-questions → Retrieves for each → Synthesizes final answer'
};

function setMode(mode, btn) {
  currentMode = mode;
  document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('mode-desc').innerHTML = MODE_DESCRIPTIONS[mode] || '';
  showToast(`Switched to ${mode.toUpperCase()} mode`);
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
  // Remove welcome message on first interaction
  const welcome = area.querySelector('.welcome');
  if (welcome) welcome.remove();

  const div = document.createElement('div');
  div.className = `message ${role}`;

  let metaHtml = '';
  if (meta.latency) metaHtml += `<span>⏱ ${meta.latency}ms</span>`;
  if (meta.confidence) metaHtml += `<span>🎯 ${Math.round(meta.confidence * 100)}%</span>`;
  if (meta.tokens) metaHtml += `<span>🔤 ${meta.tokens} tokens</span>`;
  if (meta.mode) metaHtml += `<span>🤖 ${meta.mode}</span>`;

  let sourcesHtml = '';
  if (meta.sources && meta.sources.length > 0) {
    const unique = [...new Map(meta.sources.map(s => [s.source, s])).values()];
    sourcesHtml = '<div class="sources-bar">' +
      unique.map(s => `<span class="source-badge">📄 ${s.source}</span>`).join('') +
      '</div>';
  }

  const textContent = (content || "").toString();
  div.innerHTML = `
    <div class="bubble">${textContent.replace(/\n/g, '<br>')}</div>
    ${metaHtml ? `<div class="meta">${metaHtml}</div>` : ''}
    ${sourcesHtml}
  `;
  area.appendChild(div);
  area.scrollTop = area.scrollHeight;
  return div;
}

function addThinking() {
  const area = document.getElementById('chat-area');
  const welcome = area.querySelector('.welcome');
  if (welcome) welcome.remove();
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
  const btn = document.getElementById('send-btn');
  const query = input.value.trim();
  if (!query) {
    showToast('Please type a question first!', 'error');
    input.focus();
    return;
  }

  input.value = '';
  input.style.height = 'auto';
  input.disabled = true;
  btn.disabled = true;
  
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
      
      removeThinking();
      if (!res.ok) {
        const errData = await res.json().catch(() => ({error: `Server returned ${res.status}`}));
        throw new Error(errData.error || 'Server error');
      }
      
      const data = await res.json();

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
  } finally {
    input.disabled = false;
    btn.disabled = false;
    input.focus();
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

  try {
    const res = await fetch('/api/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: query})
    });
    
    if (!res.ok) {
      const errData = await res.json().catch(() => ({error: `Server error ${res.status}`}));
      bubble.innerHTML = `❌ Error: ${errData.error}`;
      return;
    }

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
  } catch (err) {
    bubble.innerHTML = `❌ Connection Error: ${err.message}`;
  }
}

async function uploadFiles(files) {
  const bar = document.getElementById('upload-progress');
  const list = document.getElementById('sources-list');
  bar.style.width = '10%';
  list.innerHTML = '<div style="color:#6366f1;font-size:12px;padding:8px">⏳ Processing... please wait</div>';

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const formData = new FormData();
    formData.append('file', file);

    try {
      bar.style.width = `${((i + 1) / files.length) * 90}%`;
      const res = await fetch('/api/upload', {method: 'POST', body: formData});
      const data = await res.json();

      if (data.error) {
        showToast('Upload failed: ' + data.error, 'error');
        addMessage('assistant', `❌ Error uploading ${file.name}: ${data.error}`);
      } else {
        showToast(`✅ ${file.name} uploaded`);
        addMessage('assistant', `✅ Successfully loaded: ${file.name} (${data.chunks} chunks stored in Endee DB)`);
      }
    } catch (err) {
      showToast('Upload failed', 'error');
      addMessage('assistant', `❌ Connection error uploading ${file.name}: ${err.message}`);
    }
  }

  bar.style.width = '100%';
  setTimeout(() => { bar.style.width = '0'; }, 1000);
  document.getElementById('file-input').value = '';
  refreshStats();
}

async function refreshStats() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    document.getElementById('stat-vectors').textContent = data.total_vectors || 0;
    document.getElementById('stat-sources').textContent = data.unique_sources || 0;

    // Update status dot
    const statusDot = document.querySelector('.status-dot');
    if (statusDot) {
      statusDot.style.background = '#22c55e';
    }

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
      list.innerHTML = '<div style="color:#666;font-size:12px">No documents loaded yet. Upload files above.</div>';
    }
  } catch (err) {
    console.error('Stats error:', err);
    const statusDot = document.querySelector('.status-dot');
    if (statusDot) {
      statusDot.style.background = '#ef4444';
    }
    const statusText = document.querySelector('.status-text');
    if (statusText) statusText.textContent = 'Endee DB Error';
  }
}

// Initial load
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
        chunks = []
        for i, chunk in enumerate(doc.chunks):
            chunks.append({
                "id": f"{doc.doc_id}_chunk_{i}",
                "content": chunk["text"],
                "source": filename,
                "chunk_index": i
            })
        
        try:
            inserted = vector_store.add_documents(chunks)
        except ConnectionError as ce:
            return jsonify({"error": f"Endee DB not reachable: {ce}"}), 503
        
        if inserted == 0 and len(chunks) > 0:
            return jsonify({"error": "Database Error: Failed to insert vectors into Endee. Make sure Endee server is running on port 8080."}), 500
            
        return jsonify({
            "success": True,
            "filename": filename,
            "chunks": inserted
        })
    except ConnectionError as ce:
        return jsonify({"error": f"Endee DB not reachable: {ce}"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        import time
        start_time = time.time()
        
        results = vector_store.search(question, top_k=5)
        if not results:
            return jsonify({
                "answer": "No relevant documents found. Please upload documents first.",
                "sources": [],
                "latency_ms": 0,
                "confidence": 0,
                "tokens_used": 0
            })

        context = "\n\n".join([
            f"[Source: {r.source}]\n{r.content}"
            for r in results
        ])

        from groq import Groq
        client = Groq(api_key=Config.GROQ_API_KEY)
        response = client.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Answer based on the context provided. Cite sources."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            max_tokens=1024
        )
        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens
        latency_ms = int((time.time() - start_time) * 1000)

        return jsonify({
            "answer": answer,
            "sources": [{"source": r.source, "relevance_score": r.score} for r in results],
            "latency_ms": latency_ms,
            "confidence": results[0].score if results else 0,
            "tokens_used": tokens
        })
    except ConnectionError as ce:
        return jsonify({"error": f"Endee DB not reachable: {ce}"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stream", methods=["POST"])
def stream():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        def generate():
            for token in pipeline.stream_query(question):
                yield token
        return Response(stream_with_context(generate()), mimetype="text/plain")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
        try:
            vs_stats = vector_store.get_store_stats()
        except Exception:
            vs_stats = {
                "collection": Config.COLLECTION_NAME,
                "total_vectors": 0,
                "embedding_model": Config.EMBEDDING_MODEL,
                "dimension": Config.EMBEDDING_DIMENSION
            }
        
        try:
            rag_stats = pipeline.get_stats()
        except Exception:
            rag_stats = {"total_queries": 0, "total_tokens_used": 0}
            
        # Fetch uploaded sources from processor
        vs_stats["unique_sources"] = len(processor.processed_docs)
        vs_stats["sources"] = [doc.filename for doc in processor.processed_docs.values()]
        
        return jsonify({**vs_stats, **rag_stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n🚀 Starting EndeeRAG Web UI...")
    print("   Open: http://localhost:5000\n")
    app.run(debug=True, port=5000)