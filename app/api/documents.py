from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.ingestion import ingest_pdf

router = APIRouter()


@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a PDF and run the full ingestion pipeline.
    Stores document + chunks + embeddings, returns the document id.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported right now")

    file_bytes = await file.read()

    try:
        document = ingest_pdf(db, file.filename, file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "document_id": str(document.id),
        "filename": document.filename,
        "page_count": document.page_count,
        "chunk_count": document.chunk_count,
        "char_count": document.char_count,
    }