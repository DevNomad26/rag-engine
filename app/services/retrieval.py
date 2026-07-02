from sqlalchemy.orm import Session
from sqlalchemy import select
from app.core.models import Chunk
from app.services.embedder import embed_query


def retrieve_chunks(db: Session, query: str, top_k: int = 5) -> list[dict]:
    """
    Embed the query, then use pgvector to find the top_k most similar chunks
    by cosine distance — computed inside Postgres.
    """
    # 1 - embed the question
    query_vector = embed_query(query)

    # 2 - pgvector's cosine_distance does the similarity search in SQL.(semantic searching of relavent chunk)
    #    smaller distance => more similar. we order ascending and take top_k.
    stmt = (
        select(
            Chunk,
            Chunk.embedding.cosine_distance(query_vector).label("distance"),
        )
        .order_by("distance")
        .limit(top_k)
    )

    rows = db.execute(stmt).all()

    # 3 - shape results -> convert distance to a similarity score (1 - distance)
    results = []
    for chunk, distance in rows:
        results.append({
            "chunk_id": str(chunk.id),
            "document_id": str(chunk.document_id),
            "content": chunk.content,
            "chunk_index": chunk.chunk_index,
            "similarity": round(1 - distance, 4),
        })

    return results


def retrieve_chunks_by_vector(db: Session, query_vector: list[float], top_k: int = 5) -> list[dict]:
    """Retrieve by a pre-computed embedding vector (supports HyDE)."""
    stmt = (
        select(
            Chunk,
            Chunk.embedding.cosine_distance(query_vector).label("distance"),
        )
        .order_by("distance")
        .limit(top_k)
    )
    rows = db.execute(stmt).all()
    return [
        {
            "chunk_id": str(chunk.id),
            "document_id": str(chunk.document_id),
            "content": chunk.content,
            "chunk_index": chunk.chunk_index,
            "similarity": round(1 - distance, 4),
        }
        for chunk, distance in rows
    ]