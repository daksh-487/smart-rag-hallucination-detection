"""
BM25 Retriever - Keyword-based retrieval using BM25Okapi scoring.
"""

import os
import sys
import numpy as np

# Ensure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rank_bm25 import BM25Okapi

from ingestion.document_loader import load_pdfs
from ingestion.chunker import chunk_documents


class BM25Retriever:
    """
    Keyword-based retriever using BM25Okapi scoring over text chunks.
    """

    def __init__(self, chunks: list[dict]):
        """
        Builds a BM25 index from chunk dictionaries.

        Args:
            chunks: List of chunk dictionaries with "chunk_text" field.
        """
        self.chunks = chunks

        # Tokenize each chunk's text by splitting on whitespace
        self.tokenized_corpus = [
            chunk["chunk_text"].lower().split() for chunk in chunks
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"BM25 index built with {len(chunks)} chunks")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Searches the BM25 index with a query string.

        Args:
            query: The search query.
            top_k: Number of top results to return.

        Returns:
            List of chunk dictionaries sorted by BM25 score descending,
            each with an added "bm25_score" field.
        """
        # Tokenize the query
        tokenized_query = query.lower().split()

        # Get BM25 scores for all chunks
        scores = self.bm25.get_scores(tokenized_query)

        # Get top_k indices sorted by score descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build result list
        results = []
        for idx in top_indices:
            result = dict(self.chunks[idx])  # Copy original chunk fields
            result["bm25_score"] = float(scores[idx])
            results.append(result)

        return results


if __name__ == "__main__":
    # Load and chunk PDFs
    raw_folder = os.path.join(".", "data", "raw")
    documents = load_pdfs(raw_folder)
    print(f"\nPages loaded: {len(documents)}")

    chunks = chunk_documents(documents)
    print(f"Chunks created: {len(chunks)}\n")

    # Build BM25 index
    retriever = BM25Retriever(chunks)

    # Search
    query = "attention mechanism transformer"
    print(f"\nSearch query: \"{query}\"")
    print("=" * 60)

    results = retriever.search(query, top_k=5)

    for i, result in enumerate(results, 1):
        preview = result["chunk_text"][:100].replace("\n", " ")
        score = result["bm25_score"]
        source = result["source"]
        page = result["page"]

        print(f"\n  Result {i}")
        print(f"    BM25 Score : {score:.4f}")
        print(f"    Source     : {source}")
        print(f"    Page       : {page}")
        print(f"    Preview    : {preview}...")

    print(f"\n{'=' * 60}")
