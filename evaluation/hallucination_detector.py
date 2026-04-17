"""
Hallucination Detector - Uses LLM-based judge to score how faithful
an answer is to the retrieved context.
"""

import os
import sys
import json
from openai import OpenAI
import numpy as np

# Ensure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Label to score mapping
LABEL_SCORES = {
    "ENTAILMENT": 1.0,
    "NEUTRAL": 0.5,
    "CONTRADICTION": 0.0,
}


def score_faithfulness(answer: str, retrieved_chunks: list[dict], openai_client: OpenAI) -> dict:
    """
    Scores how faithful an answer is to the retrieved chunks using LLM-as-a-judge.

    Args:
        answer: The generated answer text.
        retrieved_chunks: List of chunk dicts with "chunk_text" field.
        openai_client: An initialized OpenAI client instance.

    Returns:
        Dictionary with:
            - "faithfulness_score": average score (0-1), rounded to 3 decimals
            - "sentence_scores": list of per-sentence dicts with sentence, label, score
            - "verdict": "TRUSTED", "UNCERTAIN", or "HALLUCINATED"
    """
    # Step 1: Split answer into sentences
    sentences = [s.strip() for s in answer.split(". ") if s.strip()]
    if not sentences:
        return {
            "faithfulness_score": 0.0,
            "sentence_scores": [],
            "verdict": "UNCERTAIN"
        }

    # Step 2: Extract all chunk texts for context
    context = "\n\n".join([f"Context Block {i+1}:\n{chunk.get('chunk_text', '')}" 
                          for i, chunk in enumerate(retrieved_chunks)])
    
    if not context.strip():
        context = "No context provided."

    # Step 3: Call LLM to evaluate each sentence
    # We use a structured prompt to get back JSON results for each sentence
    prompt = f"""
    You are an expert hallucination detector. Your task is to evaluate if each sentence in an "Answer" is supported by the provided "Context".
    
    For each sentence, assign one of these labels:
    - ENTAILMENT: The sentence is clearly supported by the context.
    - NEUTRAL: The context doesn't mention this information, or it's common knowledge not in the context.
    - CONTRADICTION: The sentence contradicts the information in the context.

    Context:
    {context}

    Answer:
    {answer}

    Return your results as a JSON list of objects, one for each sentence:
    [
        {{"sentence": "...", "label": "ENTAILMENT" | "NEUTRAL" | "CONTRADICTION"}},
        ...
    ]
    """

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        raw_output = response.choices[0].message.content
        data = json.loads(raw_output)
        
        # Extract the list (it might be under a key depending on LLM behavior)
        if isinstance(data, dict):
            # Try to find a list if the LLM wrapped it
            llm_results = next((v for v in data.values() if isinstance(v, list)), [])
        else:
            llm_results = data if isinstance(data, list) else []

    except Exception as e:
        print(f"Error in LLM evaluation: {e}")
        llm_results = [{"sentence": s, "label": "NEUTRAL"} for s in sentences]

    # Step 4: Map labels to scores and build final list
    sentence_scores = []
    # Match LLM results back to our sentences (fallback to NEUTRAL if mismatch)
    for i, sentence in enumerate(sentences):
        # Try to find matching sentence or use index
        label = "NEUTRAL"
        if i < len(llm_results):
            label = llm_results[i].get("label", "NEUTRAL").upper()
        
        score = LABEL_SCORES.get(label, 0.5)
        sentence_scores.append({
            "sentence": sentence,
            "label": label,
            "score": score,
        })

    # Step 5: Calculate average faithfulness score
    if sentence_scores:
        avg_score = round(float(np.mean([s["score"] for s in sentence_scores])), 3)
    else:
        avg_score = 0.0

    # Step 6: Determine verdict
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
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    print("Running LLM-based faithfulness check...")
    result = score_faithfulness(test_answer, test_chunks, client)

    print(f"FAITHFULNESS SCORE: {result['faithfulness_score']:.3f}")
    print(f"VERDICT: {result['verdict']}")
    for s in result["sentence_scores"]:
        print(f"[{s['label']}] {s['sentence']}")
