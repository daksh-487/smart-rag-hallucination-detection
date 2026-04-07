"""
Generator - Calls OpenAI API to generate answers grounded in retrieved context.
"""

import os
import sys

# Ensure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()


def generate_answer(query: str, retrieved_chunks: list[dict]) -> dict:
    """
    Generates an answer using OpenAI GPT-4o-mini grounded in retrieved chunks.

    Args:
        query: The user's question.
        retrieved_chunks: List of chunk dictionaries with "chunk_text" and "source" fields.

    Returns:
        Dictionary with "answer", "retrieved_chunks", "query", and "sources".
    """
    # Format the context from retrieved chunks
    context_parts = []
    for chunk in retrieved_chunks:
        source = chunk.get("source", "unknown")
        text = chunk.get("chunk_text", "")
        context_parts.append(f"[Source: {source}]\n{text}")

    context = "\n\n".join(context_parts)

    # Build the prompt
    prompt = (
        "You are a helpful assistant that answers questions based strictly "
        "on the provided context. If the answer is not in the context, "
        "say exactly: 'I don't have enough information to answer this.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    # Call OpenAI API
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()

    # Collect unique source filenames
    sources = list(set(chunk.get("source", "unknown") for chunk in retrieved_chunks))

    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "query": query,
        "sources": sources,
    }


if __name__ == "__main__":
    # Test with hardcoded dummy data to verify API connection
    test_query = "what is retrieval augmented generation"

    test_chunks = [
        {
            "chunk_text": (
                "Retrieval Augmented Generation (RAG) is a technique that enhances "
                "large language model outputs by first retrieving relevant documents "
                "from an external knowledge base, then using those documents as context "
                "for generating more accurate and grounded responses."
            ),
            "source": "test.pdf",
            "page": 1,
            "chunk_id": "test_page1_chunk1",
        },
        {
            "chunk_text": (
                "RAG systems typically consist of three components: a retriever that "
                "finds relevant passages, a knowledge store such as a vector database, "
                "and a generator (usually an LLM) that produces answers conditioned on "
                "the retrieved context. This reduces hallucination significantly."
            ),
            "source": "test.pdf",
            "page": 2,
            "chunk_id": "test_page1_chunk2",
        },
    ]

    print(f"Query: \"{test_query}\"")
    print(f"Chunks provided: {len(test_chunks)}")
    print("=" * 60)
    print("Calling OpenAI API...")

    try:
        result = generate_answer(test_query, test_chunks)

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources: {result['sources']}")
        print(f"\n{'=' * 60}")
        print("API connection test PASSED!")

    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nMake sure you've set OPENAI_API_KEY in your .env file!")
        print("Get your key from: https://platform.openai.com/api-keys")
