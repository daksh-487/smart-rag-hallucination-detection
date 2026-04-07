"""
Logger - Logs RAG queries, answers, and evaluation metrics to a CSV file.
"""

import csv
import os
from datetime import datetime


def log_result(query: str, answer: str, faithfulness_score: float, verdict: str, sources: list, sentence_scores: list) -> None:
    """
    Appends query results and faithfulness scores to a CSV file.

    Args:
        query: The original question
        answer: The generated answer
        faithfulness_score: The average NLI score
        verdict: TRUSTED, UNCERTAIN, or HALLUCINATED
        sources: List of source filenames used
        sentence_scores: List of dicts with sentence, label, score
    """
    csv_path = os.path.join(os.path.dirname(__file__), "..", "rag_results_log.csv")

    # Check if file exists to write header
    file_exists = os.path.isfile(csv_path)

    # Count sentence labels
    entailment_count = sum(1 for s in sentence_scores if s["label"] == "ENTAILMENT")
    neutral_count = sum(1 for s in sentence_scores if s["label"] == "NEUTRAL")
    contradiction_count = sum(1 for s in sentence_scores if s["label"] == "CONTRADICTION")

    # Format fields
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_answer = answer.replace("\n", " ").strip()
    sources_str = " | ".join(sources)
    rounded_score = round(faithfulness_score, 3)
    num_sentences = len(sentence_scores)

    # Write to CSV
    with open(csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Write header if new file
        if not file_exists:
            writer.writerow([
                "timestamp", "query", "answer", "faithfulness_score", 
                "verdict", "sources", "num_sentences", 
                "entailment_count", "neutral_count", "contradiction_count"
            ])

        # Write data row
        writer.writerow([
            timestamp, query, clean_answer, rounded_score,
            verdict, sources_str, num_sentences,
            entailment_count, neutral_count, contradiction_count
        ])

    print("✅ Result logged to rag_results_log.csv")


if __name__ == "__main__":
    # Quick test
    log_result(
        query="Test query",
        answer="This is a test answer. It has two sentences.",
        faithfulness_score=0.75,
        verdict="TRUSTED",
        sources=["test1.pdf", "test2.pdf"],
        sentence_scores=[
            {"label": "ENTAILMENT", "score": 1.0},
            {"label": "NEUTRAL", "score": 0.5}
        ]
    )
