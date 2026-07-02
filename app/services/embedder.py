import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# new google-genai client (replaces deprecated google.generativeai)
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768


def embed_query(query: str) -> list[float]:
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="retrieval_query",
            output_dimensionality=EMBED_DIM,
        ),
    )
    return result.embeddings[0].values


def embed_documents(texts: list[str]) -> list[list[float]]:
    embeddings = []
    BATCH_SIZE = 100
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type="retrieval_document",
                output_dimensionality=EMBED_DIM,
            ),
        )
        embeddings.extend([e.values for e in result.embeddings])
    return embeddings