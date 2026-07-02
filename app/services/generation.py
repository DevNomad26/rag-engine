from app.services.llm_client import chat


SYSTEM_INSTRUCTION = """You are a precise document question-answering assistant.

Rules:
- Answer ONLY using the provided context chunks below. Do not use outside knowledge.
- If the context does not contain the answer, say exactly: "The provided document does not contain enough information to answer this."
- Cite the chunks you used by their [chunk N] number inline, e.g. "The curriculum covers CNNs [chunk 2]."
- Be concise and accurate. Do not invent details not present in the context."""


def generate_answer(query: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {
            "answer": "No relevant content was found in the document for this question.",
            "citations": [],
        }

    context_block = "\n\n".join(
        f"[chunk {i + 1}]\n{c['content']}"
        for i, c in enumerate(chunks)
    )

    prompt = f"""Context chunks:

{context_block}

Question: {query}

Answer (cite chunks inline as [chunk N]):"""

    answer, usage = chat(prompt, system=SYSTEM_INSTRUCTION, return_usage=True)

    citations = [
        {
            "label": f"chunk {i + 1}",
            "chunk_id": c["chunk_id"],
            "score": c.get("rerank_score", c.get("rrf_score", c.get("similarity", 0))),
            "preview": c["content"][:160],
        }
        for i, c in enumerate(chunks)
    ]

    return {"answer": answer, "citations": citations, "usage": usage}