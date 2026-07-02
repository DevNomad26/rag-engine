import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.core.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    page_count = Column(Integer)
    char_count = Column(Integer)
    chunk_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    char_count = Column(Integer)

    # the vector column - 768 dimensions to match Google's embedding model
    embedding = Column(Vector(768))

    document = relationship("Document", back_populates="chunks")


class QueryTrace(Base):
    __tablename__ = "query_traces"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query = Column(Text, nullable=False)

    # latency in milliseconds, per stage
    retrieval_ms = Column(Integer)
    rerank_ms = Column(Integer)
    generation_ms = Column(Integer)
    total_ms = Column(Integer)

    # token usage
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)

    # config used
    used_rewriting = Column(String)  # "true"/"false" as string for simplicity

    created_at = Column(DateTime, default=datetime.utcnow)