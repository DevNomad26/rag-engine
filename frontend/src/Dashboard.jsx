import { useState, useEffect } from 'react'
import { getStats } from './api'

// Static eval results — your measured before/after story
const EVAL_RESULTS = [
  { strategy: 'Dense only',        faith: 0.959, rel: 1.000, prec: 0.600 },
  { strategy: 'Hybrid (RRF)',      faith: 0.938, rel: 0.938, prec: 0.525 },
  { strategy: 'Hybrid + Rerank',   faith: 1.000, rel: 1.000, prec: 0.525 },
]

function Bar({ label, value, max, color }) {
  const pct = max ? Math.min((value / max) * 100, 100) : 0
  return (
    <div className="dbar">
      <div className="dbar-head"><span>{label}</span><b>{value}ms</b></div>
      <div className="dbar-track"><div className="dbar-fill" style={{ width: `${pct}%`, background: color }} /></div>
    </div>
  )
}

export default function Dashboard() {
  const [stats, setStats] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    getStats().then(setStats).catch((e) => setError(e.message))
  }, [])

  return (
    <div className="dash">
      <h2 className="dash-title">Evaluation & Observability</h2>

      {/* live stat cards */}
      <div className="stat-cards">
        <div className="stat-card">
          <div className="stat-num">{stats?.total_queries ?? '—'}</div>
          <div className="stat-lbl">Total queries</div>
        </div>
        <div className="stat-card">
          <div className="stat-num">{stats ? `${stats.avg_total_ms}ms` : '—'}</div>
          <div className="stat-lbl">Avg latency</div>
        </div>
        <div className="stat-card">
          <div className="stat-num">{stats ? stats.total_tokens.toLocaleString() : '—'}</div>
          <div className="stat-lbl">Total tokens</div>
        </div>
      </div>

      {/* latency breakdown */}
      <div className="dash-panel">
        <h3>Avg latency by stage</h3>
        {stats ? (
          <>
            <Bar label="Retrieval" value={stats.avg_retrieval_ms} max={stats.avg_total_ms} color="#7c9fff" />
            <Bar label="Rerank" value={stats.avg_rerank_ms} max={stats.avg_total_ms} color="#b088ff" />
            <Bar label="Generation" value={stats.avg_generation_ms} max={stats.avg_total_ms} color="#4ade80" />
          </>
        ) : <p className="dash-empty">{error || 'Loading…'}</p>}
      </div>

      {/* eval results table */}
      <div className="dash-panel">
        <h3>Retrieval strategy evaluation <span className="dash-sub">RAGAS-style metrics · Attention Is All You Need corpus</span></h3>
        <table className="dash-table">
          <thead>
            <tr><th>Strategy</th><th>Faithfulness</th><th>Relevancy</th><th>Precision</th></tr>
          </thead>
          <tbody>
            {EVAL_RESULTS.map((r) => (
              <tr key={r.strategy}>
                <td>{r.strategy}</td>
                <td><span className="metric">{r.faith.toFixed(3)}</span></td>
                <td><span className="metric">{r.rel.toFixed(3)}</span></td>
                <td><span className="metric">{r.prec.toFixed(3)}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="dash-note">Reranking lifted faithfulness and relevancy to 1.000 by surfacing answer-bearing chunks that hybrid fusion ranked too low.</p>
      </div>

      {/* recent queries */}
      <div className="dash-panel">
        <h3>Recent queries</h3>
        {stats?.recent?.length ? (
          <div className="recent-list">
            {stats.recent.map((q, i) => (
              <div key={i} className="recent-row">
                <span className="recent-q">{q.query}</span>
                <span className="recent-meta">{q.total_ms}ms · {q.tokens} tok{q.rewriting === 'true' ? ' · HyDE' : ''}</span>
              </div>
            ))}
          </div>
        ) : <p className="dash-empty">No queries yet — ask something in Chat.</p>}
      </div>
    </div>
  )
}