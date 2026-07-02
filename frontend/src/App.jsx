import { useState } from 'react'
import { ingestDocument, askQuestion } from './api'
import Dashboard from './Dashboard'
import './App.css'

const SUGGESTIONS = [
  'Summarize this document',
  'What are the key points?',
  'What is this document about?',
]

function App() {
  const [view, setView] = useState('chat')
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState(null)
  const [question, setQuestion] = useState('')
  const [asking, setAsking] = useState(false)
  const [messages, setMessages] = useState([])
  const [useRewriting, setUseRewriting] = useState(false)

  async function handleUpload(e) {
    const file = e.target.files[0]
    if (!file) return
    setUploading(true); setUploadStatus(null)
    try {
      const r = await ingestDocument(file)
      setUploadStatus({ ok: true, text: `${r.filename} · ${r.chunk_count} chunks indexed` })
    } catch (err) {
      setUploadStatus({ ok: false, text: err.message })
    } finally { setUploading(false) }
  }

  async function ask(q) {
    if (!q.trim() || asking) return
    setQuestion('')
    setMessages((p) => [...p, { role: 'user', text: q }])
    setAsking(true)
    try {
      const r = await askQuestion(q, useRewriting)
      setMessages((p) => [...p, { role: 'assistant', text: r.answer, citations: r.citations, trace: r.trace }])
    } catch (err) {
      setMessages((p) => [...p, { role: 'assistant', text: `Error: ${err.message}` }])
    } finally { setAsking(false) }
  }

  return (
    <div className="app">
      <nav className="nav">
        <button className={view === 'chat' ? 'nav-btn active' : 'nav-btn'} onClick={() => setView('chat')}>Chat</button>
        <button className={view === 'dashboard' ? 'nav-btn active' : 'nav-btn'} onClick={() => setView('dashboard')}>Dashboard</button>
      </nav>

      {view === 'dashboard' ? (
        <Dashboard />
      ) : (
        <>
          <header className="header">
            <span className="badge">Hybrid Retrieval · Reranked</span>
            <h1>RAG Engine</h1>
            <p>Grounded document Q&A with citations and full observability</p>
          </header>

          <div className="upload">
            <div className="upload-icon">📄</div>
            <div className="upload-body">
              <div className="upload-title">Upload a document</div>
              <div className="upload-hint">PDF · extracted, chunked, embedded & indexed</div>
            </div>
            <label className="upload-btn">
              {uploading ? 'Indexing…' : 'Choose PDF'}
              <input type="file" accept=".pdf" onChange={handleUpload} disabled={uploading} hidden />
            </label>
          </div>
          {uploadStatus && (
            <div className={`upload-status ${uploadStatus.ok ? '' : 'error'}`} style={{ marginBottom: 24 }}>
              {uploadStatus.ok ? '✓' : '✕'} {uploadStatus.text}
            </div>
          )}

          <div className="chat">
            {messages.length === 0 && !asking && (
              <div className="empty">
                Upload a document, then ask anything about it.
                <div className="suggestions">
                  {SUGGESTIONS.map((s) => (
                    <span key={s} className="chip" onClick={() => ask(s)}>{s}</span>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m, i) => (
              <div key={i} className={`msg ${m.role}`}>
                <div className="bubble">{m.text}</div>
                {m.citations && m.citations.length > 0 && (
                  <details className="sources">
                    <summary className="sources-head">▸ {m.citations.length} sources used</summary>
                    {m.citations.map((c) => (
                      <div key={c.chunk_id} className="source">
                        <div className="source-top">
                          <span className="source-label">{c.label}</span>
                          <span className="source-score">relevance {c.score?.toFixed(2)}</span>
                        </div>
                        <div className="source-preview">{c.preview}…</div>
                      </div>
                    ))}
                  </details>
                )}
                {m.trace && (
                  <div className="trace-chips">
                    <span className="stat">⚡ <b>{m.trace.total_ms}ms</b> total</span>
                    <span className="stat">🔍 <b>{m.trace.retrieval_ms}ms</b> retrieval</span>
                    <span className="stat">🎯 <b>{m.trace.rerank_ms}ms</b> rerank</span>
                    <span className="stat">✍ <b>{m.trace.generation_ms}ms</b> generation</span>
                    <span className="stat"><b>{m.trace.input_tokens + m.trace.output_tokens}</b> tokens</span>
                  </div>
                )}
              </div>
            ))}

            {asking && (
              <div className="msg assistant">
                <div className="bubble" style={{ padding: 0 }}>
                  <div className="thinking"><span className="dot"></span><span className="dot"></span><span className="dot"></span></div>
                </div>
              </div>
            )}
          </div>

          <div className="input-row">
            <div className="input-inner-wrap">
              <label className="toggle">
                <input type="checkbox" checked={useRewriting} onChange={(e) => setUseRewriting(e.target.checked)} />
                <span className="toggle-track"><span className="toggle-thumb"></span></span>
                <span className="toggle-label">Rewrite vague queries (HyDE)</span>
              </label>
              <div className="input-inner">
                <input
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && ask(question)}
                  placeholder="Ask a question about your document…"
                  disabled={asking}
                />
                <button onClick={() => ask(question)} disabled={asking || !question.trim()}>Ask</button>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

export default App