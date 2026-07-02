"""
Custom LLM-as-judge evaluator.
Each metric is a focused prompt to the LLM that scores one aspect of RAG quality.
This is exactly what eval frameworks do under the hood
"""
import json
from app.services.llm_client import chat


def _judge(prompt: str) -> dict:
    """Send a scoring prompt to the judge LLM, parse its JSON response."""
    text = chat(prompt, temperature=0).strip()
    # strip markdown code fences if the model wraps its JSON
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def score_faithfulness(answer: str, contexts: list[str]) -> dict:
    """Is every claim in the answer supported by the retrieved context?"""
    context_block = "\n\n".join(contexts)
    prompt = f"""You are evaluating whether an answer is faithful to its source context.

CONTEXT:
{context_block}

ANSWER:
{answer}

Break the answer into individual factual claims. For each claim, decide if it is
supported by the CONTEXT. Then compute faithfulness = supported_claims / total_claims.

Respond with ONLY valid JSON, no markdown:
{{"total_claims": <int>, "supported_claims": <int>, "score": <float 0-1>, "unsupported": [<list of unsupported claims>]}}"""
    return _judge(prompt)


def score_answer_relevancy(question: str, answer: str) -> dict:
    """Does the answer actually address the question asked?"""
    prompt = f"""You are evaluating whether an answer addresses the question.

QUESTION: {question}
ANSWER: {answer}

Score how directly and completely the answer addresses the question, from 0 to 1.
1.0 = fully answers it, 0.0 = irrelevant or evasive.

Respond with ONLY valid JSON, no markdown:
{{"score": <float 0-1>, "reasoning": "<one sentence>"}}"""
    return _judge(prompt)


def score_context_precision(question: str, contexts: list[str]) -> dict:
    """Of the retrieved chunks, how many were actually relevant to the question?"""
    numbered = "\n\n".join(f"[chunk {i+1}]\n{c}" for i, c in enumerate(contexts))
    prompt = f"""You are evaluating retrieval quality.

QUESTION: {question}

RETRIEVED CHUNKS:
{numbered}

For each chunk, decide if it contains information useful for answering the QUESTION.
Then compute precision = useful_chunks / total_chunks.

Respond with ONLY valid JSON, no markdown:
{{"total_chunks": <int>, "useful_chunks": <int>, "score": <float 0-1>}}"""
    return _judge(prompt)