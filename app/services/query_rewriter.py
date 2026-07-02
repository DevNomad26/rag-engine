from app.services.llm_client import chat


def expand_query(query: str) -> str:
    """
    Rewrite a user's question into a more explicit, keyword-rich retrieval
    query. Helps when the original question is short or vague.
    """
    prompt = f"""Rewrite the following question into a detailed search query optimized for retrieving relevant passages from a technical document.

Add likely relevant keywords and make the intent explicit, but keep it concise (one line). Do NOT answer the question — only rewrite it as a search query.

Question: {query}

Rewritten search query:"""

    rewritten = chat(prompt, temperature=0.3).strip()
    return rewritten

#HyDE(Hypothetical Document Embedding)
def generate_hyde_document(query: str) -> str:
    """
    HyDE: generate a hypothetical answer passage for the question.
    We embed THIS instead of the raw question — a fake answer resembles
    real document chunks more than a question does, improving retrieval.
    """
    prompt = f"""Write a short, factual passage (2-3 sentences) that would directly answer the following question, as if it were an excerpt from a technical paper. Make it specific and detailed.

Question: {query}

Passage:"""

    hyde_doc = chat(prompt, temperature=0.3).strip()
    return hyde_doc