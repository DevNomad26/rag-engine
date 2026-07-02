"""
RAG evaluation harness — runs golden questions through our engine
and scores results with our custom LLM-as-judge evaluator.
"""
import time
from app.core.database import SessionLocal
from app.services.retrieval import retrieve_chunks
from app.services.generation import generate_answer
from eval.evaluator import score_faithfulness, score_answer_relevancy, score_context_precision
from app.services.hybrid_retriever import hybrid_search,hybrid_rerank_search

GOLDEN_SET = [
    {"question": "What is the name of the model architecture proposed in the paper?",
     "ground_truth": "The Transformer."},
    {"question": "What mechanism does the Transformer rely on entirely, dispensing with recurrence and convolutions?",
     "ground_truth": "Attention mechanisms, specifically self-attention."},
    {"question": "How many layers are in the encoder and decoder stacks?",
     "ground_truth": "Both encoder and decoder have 6 identical layers."},
    {"question": "What type of attention lets the model jointly attend to information from different representation subspaces?",
     "ground_truth": "Multi-head attention."},
    {"question": "What functions are used for positional encoding?",
     "ground_truth": "Sine and cosine functions of different frequencies."},
    {"question": "What BLEU score did the big Transformer achieve on WMT 2014 English-to-German?",
     "ground_truth": "28.4 BLEU."},
    {"question": "What optimizer was used to train the model?",
     "ground_truth": "The Adam optimizer."},
    {"question": "Why is self-attention preferred over recurrent layers?",
     "ground_truth": "Lower computational complexity per layer, more parallelization, shorter path length for long-range dependencies."},
]


def main():
    db = SessionLocal()
    faith_scores, rel_scores, prec_scores = [], [], []

    try:
        for i, item in enumerate(GOLDEN_SET, 1):
            q = item["question"]
            print(f"\n[{i}/{len(GOLDEN_SET)}] {q[:65]}")

            # run our real RAG pipeline
            chunks = hybrid_rerank_search(db, q, top_k=5, use_rewriting=True)  #retrieved chunks
            contexts = [c["content"] for c in chunks]
            result = generate_answer(q, chunks)
            answer = result["answer"]

            # score with our judge (delays to respect free-tier rate limits)
            faith = score_faithfulness(answer, contexts)    #check for generation(LLM - hallucination)
            time.sleep(3)
            rel = score_answer_relevancy(q, answer)         #check for generation(LLM)
            time.sleep(3)
            prec = score_context_precision(q, contexts)     #check for retrieval quality
            time.sleep(3)

            faith_scores.append(faith["score"])
            rel_scores.append(rel["score"])
            prec_scores.append(prec["score"])

            print(f"   faithfulness={faith['score']:.2f}  relevancy={rel['score']:.2f}  precision={prec['score']:.2f}")
    finally:
        db.close()

    n = len(faith_scores)
    print("\n" + "=" * 40)
    print("BASELINE RAG METRICS (averaged)")
    print("=" * 40)
    print(f"Faithfulness:      {sum(faith_scores)/n:.3f}")
    print(f"Answer Relevancy:  {sum(rel_scores)/n:.3f}")
    print(f"Context Precision: {sum(prec_scores)/n:.3f}")


if __name__ == "__main__":
    main()