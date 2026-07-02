from sqlalchemy.orm import Session
from app.services.retrieval import retrieve_chunks,retrieve_chunks_by_vector     # dense
from app.services.bm25_retriever import bm25_search                             # sparse
from app.services.reranker import rerank                                       # gives reranked top k candidates
from app.services.query_rewriter import expand_query, generate_hyde_document
from app.services.embedder import embed_query
def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Combine two ranked lists using Reciprocal Rank Fusion.

    RRF scores by RANK position, not raw score — so it sidesteps the
    problem that cosine similarity (0-1) and BM25 scores (unbounded)
    live on totally different scales. A chunk ranking high in BOTH
    lists wins; a chunk high in only one ranks medium.

    score(chunk) = sum over each list of  1 / (k + rank_in_that_list)
    """
    # map chunk_id -> its rank in each list (rank starts at 1)
    scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for rank, item in enumerate(dense_results, start=1):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunk_data[cid] = item

    for rank, item in enumerate(bm25_results, start=1):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunk_data[cid] = item  # same chunk, fine to overwrite

    # sort chunk_ids by fused score, descending
    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    # build final result list with the fused score attached
    fused = []
    for cid in ranked_ids:
        item = dict(chunk_data[cid])
        item["rrf_score"] = round(scores[cid], 5)
        fused.append(item)

    return fused


def hybrid_search(db: Session, query: str, top_k: int = 5, candidate_k: int = 10) -> list[dict]:
    """
    Run dense + BM25 retrieval, fuse with RRF, return top_k.

    candidate_k: how many to pull from each retriever before fusing.
                 We over-fetch (10) then keep the best (5) after fusion.
    """
    dense_results = retrieve_chunks(db, query, top_k=candidate_k)
    bm25_results = bm25_search(db, query, top_k=candidate_k)

    fused = reciprocal_rank_fusion(dense_results, bm25_results)
    return fused[:top_k]


def hybrid_rerank_search(
    db: Session,
    query: str,
    top_k: int = 5,
    candidate_k: int = 15,
    use_rewriting: bool = False,
) -> list[dict]:
    """
    Full pipeline with optional query rewriting:
      (rewrite) -> dense + BM25 -> RRF -> rerank -> top_k
    """
    if use_rewriting:
        # expand the query for keyword (BM25) search
        search_query = expand_query(query)
        print(f"   [rewrite] {search_query[:70]}")
        # use HyDE for dense search — embed a hypothetical answer
        hyde_doc = generate_hyde_document(query)
        dense_vector = embed_query(hyde_doc)
    else:
        search_query = query
        dense_vector = embed_query(query)

    # dense retrieval using whichever vector we chose
    dense_results = retrieve_chunks_by_vector(db, dense_vector, top_k=candidate_k)
    # BM25 using the expanded text query (gets more relavent keywords for answer)
    bm25_results = bm25_search(db, search_query, top_k=candidate_k)

    fused = reciprocal_rank_fusion(dense_results, bm25_results)
    candidates = fused[:candidate_k]

    # rerank against the ORIGINAL query (that's what the user actually asked)
    return rerank(query, candidates, top_k=top_k)