from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.core.database import get_db
from app.core.models import QueryTrace

router = APIRouter()


@router.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Live aggregate stats from the query trace log."""
    total = db.query(func.count(QueryTrace.id)).scalar() or 0

    if total == 0:
        return {
            "total_queries": 0,
            "avg_total_ms": 0, "avg_retrieval_ms": 0,
            "avg_rerank_ms": 0, "avg_generation_ms": 0,
            "total_tokens": 0, "recent": [],
        }

    avg = db.query(
        func.avg(QueryTrace.total_ms),
        func.avg(QueryTrace.retrieval_ms),
        func.avg(QueryTrace.rerank_ms),
        func.avg(QueryTrace.generation_ms),
        func.sum(QueryTrace.input_tokens + QueryTrace.output_tokens),
    ).one()

    recent = (
        db.query(QueryTrace)
        .order_by(QueryTrace.created_at.desc())
        .limit(10)
        .all()
    )

    return {
        "total_queries": total,
        "avg_total_ms": round(avg[0] or 0),
        "avg_retrieval_ms": round(avg[1] or 0),
        "avg_rerank_ms": round(avg[2] or 0),
        "avg_generation_ms": round(avg[3] or 0),
        "total_tokens": int(avg[4] or 0),
        "recent": [
            {
                "query": t.query[:60],
                "total_ms": t.total_ms,
                "tokens": (t.input_tokens or 0) + (t.output_tokens or 0),
                "rewriting": t.used_rewriting,
            }
            for t in recent
        ],
    }