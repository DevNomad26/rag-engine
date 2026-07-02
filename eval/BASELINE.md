## Results across retrieval strategies
| Strategy            | Faithfulness | Relevancy | Precision |
|---------------------|--------------|-----------|-----------|
| Dense only          | 0.959        | 1.000     | 0.600     |
| Hybrid (RRF, clean) | 0.938        | 0.938     | 0.525     |
| Hybrid + Reranking  | 1.000        | 1.000     | 0.525     |

## Query rewriting + HyDE experiment
| Config                        | Faithfulness | Relevancy | Precision |
|-------------------------------|--------------|-----------|-----------|
| Hybrid + Rerank               | 1.000        | 1.000     | 0.525     |
| Hybrid + Rerank + Rewrite/HyDE| 1.000        | 1.000     | 0.525     |

Finding: no measurable gain on single-document corpus — reranker already
saturates retrieval quality. Kept as a toggle (default off) to avoid 2 extra
LLM calls/query. Expected to help more on large heterogeneous collections.

## Latency optimization (via tracing)
Per-stage tracing revealed retrieval was 49% of total latency because
BM25 rebuilt its index on every query.

| Stage      | Before  | After caching |
|------------|---------|---------------|
| Retrieval  | 1675 ms | ~660 ms  (↓60%) |
| Total      | 3440 ms | ~3000 ms |

Fix: cached BM25 index in memory, invalidated on ingestion.
New bottleneck: cross-encoder reranking (~1500ms) — candidate for GPU/lighter model.
