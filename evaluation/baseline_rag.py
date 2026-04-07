"""
Baseline RAG - A "dumb" baseline using only basic vector search (no BM25/hybrid)
for comparison against the advanced RAG pipeline.
"""

import os
import sys

# Ensure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.document_loader import load_pdfs
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_and_store
from generation.generator import generate_answer

# Cache variables to avoid rebuilding the index on every single query
_cached_client = None
_cached_model = None
_cached_chunks = None


def run_baseline_rag(query: str) -> dict:
    """
    Runs a baseline RAG pipeline using only simple dense vector retrieval.

    Args:
        query: The user's question.

    Returns:
        Dictionary with query, answer, retrieved_chunks, and sources.
    """
    global _cached_client, _cached_model, _cached_chunks

    # 1. Load PDFs, chunk them, and build the simple vector DB (only once per session)
    if _cached_client is None:
        print("\n[Baseline RAG] Building basic vector index...")
        raw_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
        documents = load_pdfs(raw_folder)
        
        _cached_chunks = chunk_documents(documents)
        
        # embed_and_store builds an in-memory client and handles embedding 
        # using sentence-transformers all-MiniLM-L6-v2. It returns (client, model)
        _cached_client, _cached_model = embed_and_store(_cached_chunks)
        print("[Baseline RAG] Basic vector index built successfully.\n")

    # Build lookup table for chunks by chunk_id
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in _cached_chunks}

    # 2 & 3. Embed query and retrieve top 5 using ONLY cosine similarity
    query_embedding = _cached_model.encode(query).tolist()
    
    vector_results = _cached_client.query_points(
        collection_name="rag_documents",
        query=query_embedding,
        limit=5,
    )

    # Reconstruct the retrieved chunks list
    retrieved_chunks = []
    for point in vector_results.points:
        chunk_id = point.payload["chunk_id"]
        # Add the chunk with a 'vector_score' (ignoring BM25/RRF completely)
        retrieved_chunk = dict(chunk_lookup[chunk_id])
        retrieved_chunk["vector_score"] = float(point.score)
        retrieved_chunks.append(retrieved_chunk)

    # 4. Generate answer using OpenAI
    result = generate_answer(query, retrieved_chunks)

    # 5. Return the structured dictionary
    return {
        "query": query,
        "answer": result["answer"],
        "retrieved_chunks": retrieved_chunks,
        "sources": result["sources"],
    }


if __name__ == "__main__":
    # Quick test of the baseline
    test_query = "what is retrieval augmented generation"
    print(f"Testing baseline RAG with query: '{test_query}'\n")
    
    result = run_baseline_rag(test_query)
    
    print("=" * 60)
    print("BASELINE ANSWER:")
    print("=" * 60)
    print(result["answer"])
    
    print("\nRETRIEVED CHUNKS:")
    for i, c in enumerate(result["retrieved_chunks"], 1):
        print(f"  {i}. [{c['source']}] (Cosine Score: {c.get('vector_score', 0):.4f})")
    
    print("\nBaseline test completed!")
