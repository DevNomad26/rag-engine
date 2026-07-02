from sentence_transformers import CrossEncoder

# A small, fast, well-proven reranker model 
# (a large query-passage relevance dataset). Runs on CPU.
_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Reorder candidate chunks by true relevance to the query using a
    cross-encoder, then return the top_k.

    The cross-encoder reads (query, chunk) pairs together and scores how
    well each chunk answers the query — far more precise than the
    rank-based fusion that produced these candidates.
    """
    if not candidates:
        return []

    # build (query, chunk_text) pairs for the model to score
    pairs = [(query, c["content"]) for c in candidates]

    # model returns a relevance score per pair (higher = more relevant)
    scores = _model.predict(pairs)

    # attach scores and sort descending
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]