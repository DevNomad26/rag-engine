import re


def clean_extracted_text(text: str) -> str:
    """
    Clean pymupdf4llm markdown output before chunking.

    pymupdf4llm wraps text extracted from inside images/figures in
    HTML comment markers. That 'picture text' is usually OCR noise from
    diagrams (repeated tokens, <EOS>/<pad> artifacts) and hurts retrieval,
    so we remove it. We keep everything else (real prose, headings, captions).
    """
    # remove blocks between picture-text markers (the figure OCR noise)
    text = re.sub(
        r"<!-- Start of picture text -->.*?<!-- End of picture text -->",
        "",
        text,
        flags=re.DOTALL,
    )

    # strip leftover special tokens that sometimes appear outside markers
    text = re.sub(r"<EOS>|<pad>|<unk>", "", text)

    # collapse 3+ blank lines into a clean paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()