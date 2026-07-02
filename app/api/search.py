from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.core.database import get_db
from app.services.retrieval import retrieve_chunks,retrieve_chunks_by_vector
from app.services.generation import generate_answer
from app.services.bm25_retriever import bm25_search
from app.services.hybrid_retriever import hybrid_search,hybrid_rerank_search,reciprocal_rank_fusion
from app.core.models import QueryTrace
from app.services.tracing import timer
from app.services.embedder import embed_query
from app.services.reranker import rerank
from app.services.query_rewriter import expand_query, generate_hyde_document


router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/search")
async def search(req: SearchRequest, db: Session = Depends(get_db)):
    """Retrieval only - returns relevant chunks."""
    results = retrieve_chunks(db, req.query, req.top_k)
    return {"query": req.query, "results": results}


class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    use_rewriting: bool = False

@router.post("/ask")
async def ask(req: AskRequest, db: Session = Depends(get_db)):
    with timer() as retrieval_time:
        if req.use_rewriting:
            search_query = expand_query(req.query)
            hyde_doc = generate_hyde_document(req.query)
            query_vector = embed_query(hyde_doc)
        else:
            search_query = req.query
            query_vector = embed_query(req.query)

        dense = retrieve_chunks_by_vector(db, query_vector, top_k=15)
        sparse = bm25_search(db, search_query, top_k=15)
        fused = reciprocal_rank_fusion(dense, sparse)[:15]

    with timer() as rerank_time:
        chunks = rerank(req.query, fused, top_k=5)   # always rerank against original query

    with timer() as generation_time:
        result = generate_answer(req.query, chunks)

    total_ms = retrieval_time() + rerank_time() + generation_time()

    trace = QueryTrace(
        query=req.query,
        retrieval_ms=retrieval_time(), rerank_ms=rerank_time(),
        generation_ms=generation_time(), total_ms=total_ms,
        input_tokens=result["usage"]["input_tokens"],
        output_tokens=result["usage"]["output_tokens"],
        used_rewriting=str(req.use_rewriting).lower(),
    )
    db.add(trace); db.commit()

    return {
        "query": req.query,
        "answer": result["answer"],
        "citations": result["citations"],
        "trace": {**{k: getattr(trace, k) for k in
            ["retrieval_ms","rerank_ms","generation_ms","total_ms","input_tokens","output_tokens"]}},
    }
