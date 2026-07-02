import pymupdf4llm
import pymupdf
from io import BytesIO


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text as clean markdown using pymupdf4llm.
    Preserves headings, structure, and figure captions; handles
    multi-column layout far better than pypdf.
    """
    # pymupdf4llm needs a pymupdf document; open it from bytes
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    markdown_text = pymupdf4llm.to_markdown(doc)
    doc.close()
    return markdown_text


def get_pdf_metadata(file_bytes: bytes) -> dict:
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    page_count = doc.page_count
    doc.close()
    return {"page_count": page_count}