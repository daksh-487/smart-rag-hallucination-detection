"""
Hybrid Retriever - Combines BM25 keyword search with vector similarity search
using Reciprocal Rank Fusion (RRF) for optimal retrieval.
"""

import os
import sys

# Ensure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from retrieval.bm25_retriever import BM25Retriever
from ingestion.document_loader import load_pdfs
from ingestion.chunker import chunk_documents


class HybridRetriever:
    """
    Combines BM25 keyword retrieval with dense vector search
    using Reciprocal Rank Fusion (RRF) to produce final rankings.
    """

    def __init__(self, chunks: list[dict], openai_client: OpenAI):
        """
        Initializes both BM25 and vector retrieval pipelines.

        Args:
            chunks: List of chunk dictionaries from chunk_documents().
            openai_client: An initialized OpenAI client instance.
        """
        self.chunks = chunks
        self.client = openai_client
        self.embedding_model = "text-embedding-3-small"

        # Build a chunk_id -> chunk lookup for fast access
        self.chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}

        # Step 1: Create BM25 retriever
        self.bm25_retriever = BM25Retriever(chunks)

        # Step 2: Create in-memory Qdrant client and embed all chunks
        self.qdrant = QdrantClient(":memory:")
        self.collection_name = "rag_documents"

        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=1536,  # OpenAI text-embedding-3-small dimension
                distance=Distance.COSINE,
            ),
        )

        # Embed all chunks in batches
        print(f"Embedding {len(chunks)} chunks via OpenAI {self.embedding_model}...")
        texts = [chunk["chunk_text"] for chunk in chunks]
        
        all_embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.embedding_model
            )
            all_embeddings.extend([data.embedding for data in response.data])

        # Upsert into Qdrant
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
            points.append(
                PointStruct(
                    id=i,
                    vector=embedding,
                    payload={
                        "chunk_text": chunk["chunk_text"],
                        "source": chunk["source"],
                        "page": chunk["page"],
                        "chunk_id": chunk["chunk_id"],
                    },
                )
            )

        # Upsert in batches of 100
        for start in range(0, len(points), 100):
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points[start : start + 100],
            )

        print(f"HybridRetriever ready with {len(chunks)} chunks (API-driven)")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Performs hybrid search using BM25 + vector search with RRF fusion.

        Args:
            query: The search query string.
            top_k: Number of top results to return.

        Returns:
            List of chunk dicts sorted by RRF score descending,
            each with an added "rrf_score" field.
        """
        # Step 1: BM25 search — get top 20 results with ranks
        bm25_results = self.bm25_retriever.search(query, top_k=20)
        bm25_ranks = {}
        for rank, result in enumerate(bm25_results, 1):
            bm25_ranks[result["chunk_id"]] = rank

        # Step 2: Vector search — embed query and search Qdrant top 20
        response = self.client.embeddings.create(
            input=[query],
            model=self.embedding_model
        )
        query_embedding = response.data[0].embedding

        vector_results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=20,
        )
        vector_ranks = {}
        for rank, point in enumerate(vector_results.points, 1):
            vector_ranks[point.payload["chunk_id"]] = rank

        # Step 3: Reciprocal Rank Fusion (RRF)
        # Collect all unique chunk_ids from both result sets
        all_chunk_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys())

        rrf_scores = {}
        k = 60  # RRF constant

        for chunk_id in all_chunk_ids:
            bm25_rank = bm25_ranks.get(chunk_id, 999)
            vector_rank = vector_ranks.get(chunk_id, 999)
            rrf_score = 1.0 / (bm25_rank + k) + 1.0 / (vector_rank + k)
            rrf_scores[chunk_id] = rrf_score

        # Sort by RRF score descending and take top_k
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        # Build final result list
        results = []
        for chunk_id in sorted_ids:
            chunk = dict(self.chunk_lookup[chunk_id])
            chunk["rrf_score"] = rrf_scores[chunk_id]
            results.append(chunk)

        return results


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Load and chunk PDFs
    raw_folder = os.path.join(".", "data", "raw")
    documents = load_pdfs(raw_folder)
    print(f"\nPages loaded: {len(documents)}")

    chunks = chunk_documents(documents)
    print(f"Chunks created: {len(chunks)}\n")

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Create hybrid retriever
    retriever = HybridRetriever(chunks, client)

    # Test queries
    queries = [
        "what is retrieval augmented generation",
        "attention mechanism in transformers",
        "hallucination in large language models",
    ]

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"QUERY: \"{query}\"")
        print(f"{'=' * 60}")

        results = retriever.search(query, top_k=3)

        for i, result in enumerate(results, 1):
            preview = result["chunk_text"][:150].replace("\n", " ")
            print(f"\n  Result {i}")
            print(f"    Source    : {result['source']}")
            print(f"    RRF Score : {result['rrf_score']:.6f}")
            print(f"    Preview   : {preview}...")

    print(f"\n{'=' * 60}")
    print("Hybrid retrieval test complete!")
    print(f"{'=' * 60}")
