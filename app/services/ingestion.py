from sqlalchemy.orm import Session
from app.core.models import Document, Chunk
from app.services.document_parser import extract_text_from_pdf, get_pdf_metadata
from app.services.chunker import chunk_text
from app.services.embedder import embed_documents
from app.services.cleaning import clean_extracted_text
from app.services.bm25_retriever import invalidate_bm25_index

def ingest_pdf(db: Session, filename: str, file_bytes: bytes) -> Document:
    """
    Full ingestion pipeline:
      extract text -> chunk -> batch-embed -> persist document + chunks.
    Returns the saved Document.
    """
    # 1 - extract
    text = extract_text_from_pdf(file_bytes)
    text = clean_extracted_text(text)
    metadata = get_pdf_metadata(file_bytes)

    if not text.strip():
        raise ValueError("No extractable text found in this PDF.")

    # 2 - chunk
    chunks = chunk_text(text, chunk_size=800, chunk_overlap=100)

    # 3 - batch-embed all chunk contents at once
    chunk_texts = [c["content"] for c in chunks]
    embeddings = embed_documents(chunk_texts)

    # 4 - create the parent document row
    document = Document(
        filename=filename,
        page_count=metadata["page_count"],
        char_count=len(text),
        chunk_count=len(chunks),
    )
    db.add(document)
    db.flush()  # assigns document.id without committing yet

    # 5 - create chunk rows linked to the document
    for chunk_data, embedding in zip(chunks, embeddings):
        db.add(Chunk(
            document_id=document.id,
            chunk_index=chunk_data["chunk_index"],
            content=chunk_data["content"],
            char_count=chunk_data["char_count"],
            embedding=embedding,
        ))

    # 6 - commit everything in one transaction
    db.commit()
    db.refresh(document)
    #invalidate the cached index of BM25Index instance(_index)
    invalidate_bm25_index()
    return document