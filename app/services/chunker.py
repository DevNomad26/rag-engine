from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[dict]:
    """
    Splits text into overlapping chunks using LangChain's recursive splitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""], 
        length_function=len,
    )

    raw_chunks = splitter.split_text(text)

    return [
        {
            "chunk_index": i,
            "content": chunk.strip(),
            "char_count": len(chunk),
        }
        for i, chunk in enumerate(raw_chunks)
    ]