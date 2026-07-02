from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session
from app.core.models import Chunk


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Index:
    """
    In-memory BM25 index, built once and reused across queries.
    Caches plain chunk DATA (not ORM objects) so it survives session close.
    """
    def __init__(self):
        self._bm25 = None
        self._chunk_data = None  # list of plain dicts, session-independent

    def build(self, db: Session):
        chunks = db.query(Chunk).all()
        # never cache the ORM objects themselves - they die with the session.
        self._chunk_data = [
            {
                "chunk_id": str(c.id),
                "document_id": str(c.document_id),
                "content": c.content,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]
        if self._chunk_data:
            corpus = [_tokenize(c["content"]) for c in self._chunk_data]
            self._bm25 = BM25Okapi(corpus)
        else:
            self._bm25 = None

    def invalidate(self):
        self._bm25 = None
        self._chunk_data = None

    def is_built(self) -> bool:
        return self._bm25 is not None

    def search(self, db: Session, query: str, top_k: int = 10) -> list[dict]:
        if not self.is_built():
            self.build(db)

        if not self._bm25 or not self._chunk_data:
            return []

        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        scored = sorted(zip(self._chunk_data, scores), key=lambda p: p[1], reverse=True)
        top = scored[:top_k]

        return [
            {**chunk_dict, "bm25_score": round(float(score), 4)}
            for chunk_dict, score in top
        ]


_index = BM25Index()


def bm25_search(db: Session, query: str, top_k: int = 10) -> list[dict]:
    return _index.search(db, query, top_k)


def invalidate_bm25_index():
    _index.invalidate()