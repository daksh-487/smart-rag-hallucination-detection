"""
Embedder - Embeds text chunks using sentence-transformers and stores them in Qdrant (in-memory).
"""

import os
import sys

# Ensure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from ingestion.document_loader import load_pdfs
from ingestion.chunker import chunk_documents


def embed_and_store(chunks: list[dict], client: QdrantClient = None) -> QdrantClient:
    """
    Embeds chunk texts and stores them in an in-memory Qdrant collection.

    Args:
        chunks: List of chunk dictionaries from chunk_documents().
        client: Optional existing QdrantClient. Creates in-memory client if None.

    Returns:
        The QdrantClient instance with all vectors stored.
    """
    # Step 1: Load the embedding model
    print("Loading embedding model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 2: Embed all chunk texts in batches of 32
    print(f"Embedding {len(chunks)} chunks...")
    texts = [chunk["chunk_text"] for chunk in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    # Step 3: Create in-memory Qdrant client if not provided
    if client is None:
        client = QdrantClient(":memory:")

    # Step 4: Create collection (skip if already exists)
    collection_name = "rag_documents"
    existing_collections = [c.name for c in client.get_collections().collections]

    if collection_name not in existing_collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=384,  # all-MiniLM-L6-v2 output dimension
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection '{collection_name}'")
    else:
        print(f"Collection '{collection_name}' already exists, skipping creation")

    # Step 5: Upsert all chunks into Qdrant
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point = PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={
                "chunk_text": chunk["chunk_text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
            },
        )
        points.append(point)

    # Upsert in batches of 100 to avoid large payloads
    batch_size = 100
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=collection_name, points=batch)

    # Step 6: Print confirmation
    print(f"Stored {len(points)} vectors in Qdrant")

    # Step 7: Return the client
    return client, model


if __name__ == "__main__":
    # Load PDFs
    raw_folder = os.path.join(".", "data", "raw")
    documents = load_pdfs(raw_folder)
    print(f"\nPages loaded: {len(documents)}")

    # Chunk documents
    chunks = chunk_documents(documents)
    print(f"Chunks created: {len(chunks)}\n")

    # Embed and store
    client, model = embed_and_store(chunks)

    # Test search
    print("\n" + "=" * 60)
    print("TEST SEARCH: 'what is retrieval augmented generation'")
    print("=" * 60)

    query = "what is retrieval augmented generation"
    query_embedding = model.encode(query).tolist()

    results = client.query_points(
        collection_name="rag_documents",
        query=query_embedding,
        limit=3,
    )

    for i, point in enumerate(results.points, 1):
        print(f"\n--- Result {i} ---")
        print(f"  Score  : {point.score:.4f}")
        print(f"  Source : {point.payload['source']}")
        print(f"  Page   : {point.payload['page']}")
        print(f"  ChunkID: {point.payload['chunk_id']}")
        print(f"  Text   : {point.payload['chunk_text'][:200]}...")

