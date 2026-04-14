"""
Smart RAG System - Complete pipeline: Load → Chunk → Retrieve → Generate
"""

import os
import sys
import pandas as pd

from sentence_transformers import SentenceTransformer

from ingestion.document_loader import load_pdfs
from ingestion.chunker import chunk_documents
from retrieval.hybrid_retriever import HybridRetriever
from generation.generator import generate_answer
from evaluation.hallucination_detector import score_faithfulness
from evaluation.logger import log_result


def run_rag(query: str, retriever: HybridRetriever) -> dict:
    """
    Runs the full RAG pipeline for a given query using a pre-built retriever.

    Args:
        query: The user's question.
        retriever: A pre-built HybridRetriever instance.

    Returns:
        The result dictionary from generate_answer.
    """
    # Step 1: Retrieve top 5 chunks
    retrieved_chunks = retriever.search(query, top_k=5)

    # Step 2: Generate answer using OpenAI
    result = generate_answer(query, retrieved_chunks)

    # Step 3: Run hallucination detection
    detection_result = score_faithfulness(result["answer"], retrieved_chunks)

    # Step 4: Print results
    print(f"\n{'=' * 70}")
    print(f"QUERY: {result['query']}")
    print(f"{'=' * 70}")

    print(f"\nANSWER:\n{result['answer']}")

    print("\n----------------------------------------")
    print(f"FAITHFULNESS SCORE: {detection_result['faithfulness_score']:.3f}")
    print(f"VERDICT: {detection_result['verdict']}")
    print("----------------------------------------")
    
    verdict = detection_result["verdict"]
    if verdict == "HALLUCINATED":
        print("[WARNING] This answer may not be supported by your documents. Please verify manually.")
    elif verdict == "UNCERTAIN":
        print("[NOTE] This answer is partially supported. Cross-check with the source documents below.")
    elif verdict == "TRUSTED":
        print("[SUCCESS] This answer is well supported by the retrieved documents.")

    print(f"\nSOURCES: {result['sources']}")

    print("\nSENTENCE BREAKDOWN:")
    for s in detection_result["sentence_scores"]:
        print(f"  [{s['label']}] {s['sentence']}")

    print(f"\nCHUNKS USED:")
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk["source"]
        rrf_score = chunk.get("rrf_score", 0)
        preview = chunk["chunk_text"][:100].replace("\n", " ")
        print(f"  {i}. [{source}] (RRF: {rrf_score:.6f}) {preview}...")

    print("\n----------------------------------------")
    log_result(
        query=query,
        answer=result["answer"],
        faithfulness_score=detection_result["faithfulness_score"],
        verdict=detection_result["verdict"],
        sources=result["sources"],
        sentence_scores=detection_result["sentence_scores"]
    )
    print("----------------------------------------")

    return {
        "query": query,
        "answer": result["answer"],
        "retrieved_chunks": retrieved_chunks,
        "sources": result["sources"],
        "faithfulness_score": detection_result["faithfulness_score"],
        "verdict": detection_result["verdict"],
        "sentence_scores": detection_result["sentence_scores"],
    }


def build_pipeline():
    """
    Builds the RAG pipeline: loads PDFs, chunks them, and creates the retriever.
    Returns the HybridRetriever ready for queries.
    """
    # Step 1: Load PDFs
    raw_folder = os.path.join(".", "data", "raw")
    documents = load_pdfs(raw_folder)
    print(f"\nPages loaded: {len(documents)}")

    # Step 2: Chunk documents
    chunks = chunk_documents(documents)
    print(f"Chunks created: {len(chunks)}\n")

    # Step 3: Load embedding model
    print("Loading embedding model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 4: Create hybrid retriever
    retriever = HybridRetriever(chunks, model)

    return retriever


if __name__ == "__main__":
    print("RAG system starting...")
    print("=" * 70)

    # Build the pipeline once (expensive step)
    retriever = build_pipeline()

    # Run 5 test queries
    queries = [
        "what is retrieval augmented generation",
        "how do transformers handle long context windows",
        "what are the main causes of hallucination in LLMs",
        "explain the difference between dense and sparse retrieval",
        "what is the role of attention in neural networks"
    ]

    for query in queries:
        run_rag(query, retriever)

    print(f"\n{'=' * 70}")
    print("RAG pipeline complete!")
    print(f"{'=' * 70}")

    print("\n--- RESULTS SUMMARY ---")
    log_path = os.path.join(os.path.dirname(__file__), "rag_results_log.csv")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        recent_df = df.tail(5).copy()
        recent_df["query_truncated"] = recent_df["query"].str[:50]
        
        print("\nLast 5 Queries:")
        print(recent_df[["query_truncated", "faithfulness_score", "verdict"]].to_string(index=False))
        
        avg_score = recent_df["faithfulness_score"].mean()
        print(f"\nAverage Faithfulness Score (Last 5): {avg_score:.3f}")
    else:
        print("No log file found.")
