---
title: RAG Engine
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# RAG Engine — Grounded Document Q&A with Hybrid Retrieval, Reranking & Evaluation

A production-style Retrieval-Augmented Generation system that answers questions over your documents with **grounded, cited answers** — built from primitives (no LangChain orchestration) so every layer is inspectable and tunable. Ships with a **hybrid retrieval pipeline**, **cross-encoder reranking**, and a **custom evaluation + observability layer** that measures retrieval quality and per-stage latency.

> Most RAG projects stop at "it returns an answer." This one measures whether the answer is actually grounded, tracks where every millisecond goes, and proves each optimization with before/after metrics.

---

## Demo

![Chat interface](docs/chat.png)
![Dashboard](docs/dashboard.png)

*(Add screenshots of the chat interface and the evaluation dashboard here.)*

---

## What it does

- **Ingests PDFs** with layout-aware extraction (headings, structure, and figure captions preserved), then cleans figure-OCR noise, chunks semantically, embeds, and indexes into a vector store.
- **Answers questions** using a multi-stage retrieval pipeline and a grounded LLM prompt that cites its sources inline and refuses to answer beyond the provided context (anti-hallucination).
- **Measures itself** — a custom LLM-as-judge eval harness scores faithfulness, answer relevancy, and context precision; per-query tracing logs stage-level latency and token cost.
- **Visualizes** everything in a live dashboard: retrieval-strategy comparison, latency breakdown, and query volume.

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
   PDF upload  ──▶  │  Ingestion pipeline                         │
                    │  extract (pymupdf4llm) → clean → chunk       │
                    │  → embed (Gemini, 768d) → store (pgvector)   │
                    └─────────────────────────────────────────────┘
                                        │
   Question  ──────────────────────────┼───────────────────────────┐
                                        ▼                           │
        ┌──────────────────────────────────────────────────┐       │
        │  Retrieval pipeline                               │       │
        │                                                   │       │
        │   (optional) Query rewriting + HyDE               │       │
        │        │                                          │       │
        │        ├──▶ Dense search  (pgvector cosine)       │       │
        │        └──▶ Sparse search (BM25, cached index)    │       │
        │                    │                              │       │
        │             RRF fusion (rank-based)               │       │
        │                    │                              │       │
        │        Cross-encoder reranking (top-k)            │       │
        └──────────────────────────────────────────────────┘       │
                             │                                      │
                             ▼                                      │
        ┌──────────────────────────────────────────────────┐       │
        │  Grounded generation (Groq / Llama 3.3 70B)       │       │
        │  answer + inline [chunk N] citations              │       │
        └──────────────────────────────────────────────────┘       │
                             │                                      │
                             ▼                                      ▼
                    Answer + citations               Per-query trace logged
                                                     (latency, tokens) → Dashboard
```

Every stage is a small, single-responsibility module with a clean interface — the extraction engine, retriever, and LLM provider can each be swapped without touching the rest of the pipeline.

---

## Retrieval pipeline (the core)

The system layers four techniques, each targeting a specific weakness of the previous:

**1. Dense retrieval** — semantic search over Gemini embeddings via pgvector cosine distance. Strong at meaning, weak at exact terms.

**2. Sparse retrieval (BM25)** — keyword search that catches exact names, numbers, and rare terms that embeddings blur. The BM25 index is **cached in memory** and rebuilt only on ingestion (see [Performance](#performance)).

**3. Reciprocal Rank Fusion (RRF)** — combines the dense and sparse rankings by *rank position* rather than raw score, sidestepping the problem that cosine similarity (0–1) and BM25 scores (unbounded) live on incomparable scales.

**4. Cross-encoder reranking** — a `ms-marco-MiniLM` cross-encoder reads each (question, chunk) pair *together* and rescoring by true relevance, then keeps the top-k. This is what raised faithfulness and relevancy to a perfect score (see [Evaluation](#evaluation)).

---

## Query Rewriting & Hypothetical Document Embeddings (HyDE)

The pipeline includes an **optional query-transformation layer**, toggleable per request:

- **Query expansion** — the LLM rewrites a short or vague user question into a keyword-rich retrieval query before searching, making intent explicit for the sparse (BM25) retriever.
- **HyDE (Hypothetical Document Embeddings)** — instead of embedding the *question*, the LLM writes a hypothetical *answer passage*, and that is embedded for dense search. The insight: a hypothetical answer resembles real document chunks far more than a question does, so it retrieves closer matches.

Importantly, retrieval uses the rewritten query, but **reranking always scores against the original user question** — the rewrite helps *find* candidates; the reranker judges relevance to what was actually asked.

**Honest finding (measured, not assumed):** on a single, clean, well-structured document, query rewriting + HyDE produced **no measurable gain** — the cross-encoder reranker already saturates retrieval quality, and rewriting adds two extra LLM calls (visible as higher retrieval latency in the trace). It is therefore implemented but **defaulted off**, exposed as a UI toggle. It is expected to help more on large, heterogeneous corpora where the initial candidate fetch is the bottleneck. This is a deliberate, evidence-based engineering decision rather than cargo-culting a popular technique.

---

## Evaluation

A custom **LLM-as-judge** evaluation harness (no external eval framework) scores the pipeline on a hand-curated golden set from the *Attention Is All You Need* paper. Metrics:

- **Faithfulness** — is every claim in the answer supported by the retrieved context? (hallucination detection)
- **Answer relevancy** — does the answer address the question asked?
- **Context precision** — of the retrieved chunks, how many were actually relevant? (retrieval signal-to-noise)

### Results across retrieval strategies

| Strategy            | Faithfulness | Answer Relevancy | Context Precision |
|---------------------|:------------:|:----------------:|:-----------------:|
| Dense only          | 0.959        | 1.000            | 0.600             |
| Hybrid (RRF)        | 0.938        | 0.938            | 0.525             |
| **Hybrid + Rerank** | **1.000**    | **1.000**        | 0.525             |

**Reading the results:** pure hybrid initially *regressed* — sparse retrieval pulled in figure-extraction noise that polluted the fused candidate set. After a cleaning pass (stripping figure-OCR artifacts) and adding cross-encoder reranking, faithfulness and relevancy reached a perfect 1.000 by surfacing answer-bearing chunks that fusion alone ranked too low. Building the eval harness *first* made every subsequent optimization measurable rather than guesswork.

---

## Performance

Per-stage latency tracing revealed that **retrieval was 49% of total response time** — because the BM25 index was being rebuilt from scratch on every query.

| Stage      | Before  | After caching   |
|------------|--------:|----------------:|
| Retrieval  | 1675 ms | ~660 ms (↓ 60%) |
| Total      | 3440 ms | ~3000 ms        |

**Fix:** cached the BM25 index in memory, invalidated on ingestion. After this, cross-encoder reranking (~1.4s, CPU) became the dominant cost — a candidate for GPU acceleration or a lighter rerank model. Identifying the bottleneck with real trace data (rather than guessing) is the point of the observability layer.

---

## Tech stack

| Layer            | Choice                                                  |
|------------------|---------------------------------------------------------|
| Backend          | FastAPI (Python)                                        |
| Database         | PostgreSQL + pgvector (Supabase), SQLAlchemy + Alembic  |
| PDF extraction   | pymupdf4llm (layout-aware markdown)                     |
| Embeddings       | Google Gemini `gemini-embedding-001` (768-dim)          |
| Generation       | Groq — Llama 3.3 70B (OpenAI-compatible)                |
| Sparse retrieval | BM25 (`rank-bm25`), cached                              |
| Reranking        | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, CPU)     |
| Frontend         | React + Vite                                            |
| Evaluation       | Custom LLM-as-judge harness                             |

**Provider-agnostic LLM layer:** generation and judging route through a single OpenAI-compatible client, so swapping providers is a two-line change. Embeddings stay on Gemini (generous free embedding quota); generation runs on Groq for speed and daily-limit headroom.

---

## Getting started

### Prerequisites
- Python 3.12+
- Node.js 18+
- A [Supabase](https://supabase.com) project with the `vector` extension enabled
- Free API keys: [Google AI Studio](https://aistudio.google.com/apikey) (embeddings) and [Groq](https://console.groq.com) (generation)

### Backend

```bash
# clone and enter
git clone https://github.com/DevNomad26/rag-engine.git
cd rag-engine

# virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# dependencies
pip install -r requirements.txt

# environment — create a .env file:
cat > .env << 'EOF'
GOOGLE_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
DATABASE_URL=your_supabase_session_pooler_url
EOF

# apply database migrations
alembic upgrade head

# run
uvicorn main:app --reload
```

Backend runs at `http://localhost:8000` — interactive API docs at `/docs`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173` (proxies API calls to the backend).

### Usage
1. Open the frontend, upload a PDF — it's extracted, chunked, embedded, and indexed.
2. Ask questions in the Chat tab — get grounded answers with clickable source citations and a live latency/token trace.
3. Open the Dashboard tab to see retrieval-strategy metrics and live observability stats.

---

## Project structure

```
rag-engine/
├── app/
│   ├── api/            # FastAPI routes (documents, search, dashboard)
│   ├── core/           # DB connection, SQLAlchemy models
│   └── services/       # ingestion, chunking, embedding, retrieval,
│                       #   BM25, RRF fusion, reranking, query rewriting,
│                       #   generation, tracing
├── eval/               # LLM-as-judge evaluation harness + golden set
├── alembic/            # database migrations
├── frontend/           # React + Vite UI (chat + dashboard)
└── main.py             # app entry point
```

---

## Design decisions

- **Built from primitives, not LangChain.** Chunking, fusion, reranking orchestration, and the eval harness are implemented directly so every layer is inspectable and tunable — essential for the evaluation-driven workflow this project is built around. Solved infrastructure (PDF parsing, embeddings, BM25, vector search) uses libraries.
- **Evaluation before optimization.** The eval harness was built first to establish a baseline, so every subsequent change (hybrid, reranking, caching) could be measured rather than assumed.
- **Measured, not assumed.** Query rewriting was implemented, measured, found not to help on this corpus, and correctly defaulted off — an evidence-based decision.

---

## Known limitations & future work

- In-memory BM25 cache is per-process; a multi-instance deployment would need a shared index (Redis/persisted).
- Diagram *content* (not captions) is currently dropped; a multimodal pass (vision-model diagram description) would recover it.
- Single-document focus; multi-document and cross-document querying are natural extensions.
- Evaluation golden set is hand-curated and small; scaling it (auto-generation) would strengthen the metrics' statistical weight.