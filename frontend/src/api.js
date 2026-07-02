const BASE = '/api'

export async function ingestDocument(file) {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`${BASE}/documents/ingest`, {
    method: 'POST',
    body: formData,
  })
  if (!res.ok) throw new Error(`Ingest failed: ${res.status}`)
  return res.json()
}

export async function askQuestion(query, useRewriting = false) {
  const res = await fetch(`${BASE}/search/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: 5, use_rewriting: useRewriting }),
  })
  if (!res.ok) throw new Error(`Ask failed: ${res.status}`)
  return res.json()
}

export async function getStats() {
  const res = await fetch(`${BASE}/dashboard/stats`)
  if (!res.ok) throw new Error(`Stats failed: ${res.status}`)
  return res.json()
}