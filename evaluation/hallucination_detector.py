"""
Hallucination Detector - Uses NLI model to score how faithful
an answer is to the retrieved context.
"""

import os
import sys

# Ensure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import pipeline
import numpy as np

# We initialize lazily to avoid Windows multiprocessing deadlock
nli_model = None

# Label to score mapping
LABEL_SCORES = {
    "ENTAILMENT": 1.0,
    "NEUTRAL": 0.5,
    "CONTRADICTION": 0.0,
}


def score_faithfulness(answer: str, retrieved_chunks: list[dict]) -> dict:
    """
    Scores how faithful an answer is to the retrieved chunks using NLI.

    Args:
        answer: The generated answer text.
        retrieved_chunks: List of chunk dicts with "chunk_text" field.

    Returns:
        Dictionary with:
            - "faithfulness_score": average score (0-1), rounded to 3 decimals
            - "sentence_scores": list of per-sentence dicts with sentence, label, score
            - "verdict": "TRUSTED", "UNCERTAIN", or "HALLUCINATED"
    """
    # Lazy load the model on first request
    global nli_model
    if nli_model is None:
        print("Lazy-loading NLI model 'cross-encoder/nli-deberta-v3-small'...")
        nli_model = pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small",
            device=-1,
        )

    # Step 1: Split answer into sentences
    sentences = [s.strip() for s in answer.split(". ") if s.strip()]

    # Step 2: Extract all chunk texts
    chunk_texts = [chunk.get("chunk_text", "") for chunk in retrieved_chunks]
    if not chunk_texts:
        chunk_texts = [""] # Fallback if no chunks

    # Step 3: Score each sentence against chunks individually
    sentence_scores = []

    for sentence in sentences:
        max_score = -1.0
        best_label = "NEUTRAL"

        for context in chunk_texts:
            # Format input for NLI model
            nli_input = f"{context} [SEP] {sentence}"

            # Run NLI classification
            result = nli_model(nli_input, truncation=True)
            label = result[0]["label"]
            score = LABEL_SCORES.get(label, 0.5)

            if score > max_score:
                max_score = score
                best_label = label

        sentence_scores.append({
            "sentence": sentence,
            "label": best_label,
            "score": max_score,
        })

    # Step 4: Calculate average faithfulness score
    if sentence_scores:
        avg_score = round(float(np.mean([s["score"] for s in sentence_scores])), 3)
    else:
        avg_score = 0.0

    # Step 5: Determine verdict
    if avg_score >= 0.7:
        verdict = "TRUSTED"
    elif avg_score >= 0.4:
        verdict = "UNCERTAIN"
    else:
        verdict = "HALLUCINATED"

    return {
        "faithfulness_score": avg_score,
        "sentence_scores": sentence_scores,
        "verdict": verdict,
    }


if __name__ == "__main__":
    # Test case
    test_answer = (
        "RAG combines retrieval with generation. "
        "It was invented in 2025 by Microsoft."
    )

    test_chunks = [
        {
            "chunk_text": (
                "Retrieval Augmented Generation was proposed by Lewis et al. "
                "in 2020 at Facebook AI Research. It combines a retrieval "
                "system with a sequence to sequence model."
            )
        }
    ]

    result = score_faithfulness(test_answer, test_chunks)

    print(f"FAITHFULNESS SCORE: {result['faithfulness_score']:.3f}")
    print(f"VERDICT: {result['verdict']}")
    for s in result["sentence_scores"]:
        print(f"[{s['label']}] {s['sentence']}")

    print("\n" + "=" * 60)
    print("TESTING FULL RAG INTEGRATION")
    print("=" * 60)

    # Import the pipeline from main.py
    from main import build_pipeline, run_rag

    # 1. Build the retriever
    retriever = build_pipeline()

    # 2. Run a real query
    real_query = "what is retrieval augmented generation"
    print(f"\nRunning real query: '{real_query}'")
    rag_result = run_rag(real_query, retriever)

    # 3. Score the real generated answer
    print("\n" + "=" * 60)
    print("SCORING REAL GENERATED ANSWER")
    print("=" * 60)

    real_answer = rag_result["answer"]
    retrieved_chunks = rag_result["retrieved_chunks"]

    real_score_result = score_faithfulness(real_answer, retrieved_chunks)

    print(f"\nREAL ANSWER:\n{real_answer}\n")
    print(f"FAITHFULNESS SCORE: {real_score_result['faithfulness_score']:.3f}")
    print(f"VERDICT: {real_score_result['verdict']}")

    print("\nPer-sentence breakdown:")
    for i, s in enumerate(real_score_result["sentence_scores"], 1):
        print(f"  {i}. [{s['label']}] (score: {s['score']}) {s['sentence'][:100]}...")

    print(f"\n{'=' * 60}")
