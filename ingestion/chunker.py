"""
Chunker - Splits page-level documents into smaller overlapping text chunks.
"""

import os
import sys

# Ensure the project root is in the path so imports work
# whether run as `python ingestion/chunker.py` or `python -m ingestion.chunker`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.document_loader import load_pdfs


def chunk_documents(documents: list[dict], chunk_size: int = 512, overlap: int = 50) -> list[dict]:
    """
    Splits each page text into overlapping chunks.

    Args:
        documents: List of page dictionaries from load_pdfs().
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A list of dictionaries, each with:
            - "chunk_text": the chunk content
            - "source": original PDF filename
            - "page": original page number
            - "chunk_id": unique string formatted as "filename_page2_chunk3"
    """
    chunks = []

    for doc in documents:
        text = doc["text"]
        source = doc["source"]
        page = doc["page"]

        # Remove .pdf extension for chunk_id prefix
        base_name = source.replace(".pdf", "")

        start = 0
        chunk_index = 1

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            # Skip empty or whitespace-only chunks
            if chunk_text.strip():
                chunk_id = f"{base_name}_page{page}_chunk{chunk_index}"

                chunks.append({
                    "chunk_text": chunk_text,
                    "source": source,
                    "page": page,
                    "chunk_id": chunk_id,
                })

                chunk_index += 1

            # Move forward by (chunk_size - overlap)
            start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    raw_folder = os.path.join(".", "data", "raw")

    # Step 1: Load PDFs
    documents = load_pdfs(raw_folder)
    print(f"Pages loaded: {len(documents)}\n")

    # Step 2: Chunk documents
    chunks = chunk_documents(documents)
    print(f"Total chunks created: {len(chunks)}\n")

    # Step 3: Print first chunk for verification
    if chunks:
        first = chunks[0]
        print("=" * 60)
        print("FIRST CHUNK:")
        print("=" * 60)
        print(f"  chunk_id : {first['chunk_id']}")
        print(f"  source   : {first['source']}")
        print(f"  page     : {first['page']}")
        print(f"  length   : {len(first['chunk_text'])} chars")
        print("-" * 60)
        print(first["chunk_text"])
        print("=" * 60)
