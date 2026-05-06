 123"""
Smart RAG API - Per-session RAG pipeline.
Upload a PDF → immediate ingestion/chunking/embedding → query against it.
Each upload resets the entire pipeline (fresh session every time).
"""

import csv
import os
import sys
import shutil
import glob
import fitz  # PyMuPDF

from openai import OpenAI
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Pipeline imports
from ingestion.chunker import chunk_documents
from retrieval.hybrid_retriever import HybridRetriever
from generation.generator import generate_answer
from evaluation.hallucination_detector import score_faithfulness
from evaluation.logger import log_result

# ── App setup ──────────────────────────────────────────────────────────

app = FastAPI(title="Smart RAG API", description="Backend for the Smart RAG dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ───────────────────────────────────────────────────────

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
retriever = None          # HybridRetriever built after upload
active_filename = None    # Name of the currently loaded PDF
active_chunk_count = 0    # Number of chunks in the current index
active_page_count = 0     # Number of pages in the current PDF

# ── Helpers ────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("VERCEL"):
    UPLOAD_DIR = "/tmp/uploads"
else:
    UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")


def _clear_upload_dir():
    """Wipe and recreate the uploads folder so we only ever have one PDF."""
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def _load_single_pdf(filepath: str) -> list[dict]:
    """Load a single PDF into page-level dicts, same format as document_loader."""
    documents = []
    filename = os.path.basename(filepath)
    try:
        pdf = fitz.open(filepath)
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text()
            documents.append({
                "text": text,
                "source": filename,
                "page": page_num + 1,
            })
        pdf.close()
        print(f"Loaded: {filename} ({len(documents)} pages)")
    except Exception as e:
        print(f"Error reading {filename}: {e}")
    return documents


def _build_pipeline_for_file(filepath: str):
    """
    Full RAG ingestion for a single PDF:
    load → chunk → BM25 + vector embed → HybridRetriever
    """
    global retriever, active_filename, active_chunk_count, active_page_count

    # 1. Load PDF pages
    documents = _load_single_pdf(filepath)
    active_page_count = len(documents)
    print(f"Pages loaded: {active_page_count}")

    if active_page_count == 0:
        raise ValueError("PDF produced zero pages — it may be image-only or corrupt.")

    # 2. Chunk
    chunks = chunk_documents(documents)
    active_chunk_count = len(chunks)
    print(f"Chunks created: {active_chunk_count}")

    # 3. Build hybrid retriever (BM25 + Qdrant vector store)
    retriever = HybridRetriever(chunks, openai_client)
    active_filename = os.path.basename(filepath)

    print(f"Pipeline ready for '{active_filename}' "
          f"({active_page_count} pages, {active_chunk_count} chunks)")


def _run_query(query: str) -> dict:
    """Run the full RAG pipeline: retrieve → generate → detect hallucinations."""
    if retriever is None:
        raise HTTPException(status_code=400, detail="No document loaded. Upload a PDF first.")

    # 1. Retrieve top 5 chunks
    retrieved_chunks = retriever.search(query, top_k=5)

    # 2. Generate answer
    result = generate_answer(query, retrieved_chunks)

    # 3. Hallucination detection
    detection = score_faithfulness(result["answer"], retrieved_chunks, openai_client)

    # 4. Console log (encode-safe)
    try:
        print(f"\nQUERY: {query}")
        print(f"VERDICT: {detection['verdict']}  SCORE: {detection['faithfulness_score']:.3f}")
    except UnicodeEncodeError:
        pass

    # 5. CSV log
    log_result(
        query=query,
        answer=result["answer"],
        faithfulness_score=detection["faithfulness_score"],
        verdict=detection["verdict"],
        sources=result["sources"],
        sentence_scores=detection["sentence_scores"],
    )

    return {
        "query": query,
        "answer": result["answer"],
        "retrieved_chunks": retrieved_chunks,
        "sources": result["sources"],
        "faithfulness_score": detection["faithfulness_score"],
        "verdict": detection["verdict"],
        "sentence_scores": detection["sentence_scores"],
    }


# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Returns the index.html file from the /static folder."""
    static_file = os.path.join(BASE_DIR, "static", "index.html")
    if os.path.exists(static_file):
        with open(static_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<h1>Smart RAG API</h1><p>index.html not found in /static folder</p>",
        status_code=404,
    )


@app.get("/status")
def get_status():
    """Returns system status including currently loaded document info."""
    return {
        "status": "online",
        "document": active_filename or "None",
        "pages": active_page_count,
        "chunks": active_chunk_count,
        "indexed": retriever is not None,
        "model": "GPT-4o-mini",
        "retrieval": "Hybrid RRF (BM25 + OpenAI Embeddings)",
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF → clears previous session → runs full ingestion pipeline.
    Returns chunk/page counts so the frontend can update immediately.
    """
    global retriever, active_filename, active_chunk_count, active_page_count

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # 1. Clear everything from previous session
    retriever = None
    active_filename = None
    active_chunk_count = 0
    active_page_count = 0
    _clear_upload_dir()

    # 2. Save uploaded file
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 3. Run full pipeline: load → chunk → embed → index
    try:
        _build_pipeline_for_file(filepath)
    except Exception as e:
        print(f"PIPELINE BUILD FAILED: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

    return {
        "filename": active_filename,
        "pages": active_page_count,
        "chunks": active_chunk_count,
        "status": "indexed",
    }


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def handle_query(request: QueryRequest):
    """
    Core RAG endpoint. Retrieves chunks, generates answer,
    runs hallucination detection, returns structured data.
    """
    if retriever is None:
        raise HTTPException(
            status_code=400,
            detail="No document loaded. Please upload a PDF first.",
        )

    try:
        result = _run_query(request.query)
    except HTTPException:
        raise
    except Exception as e:
        print(f"QUERY FAILED: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

    # Format sources for frontend
    formatted_sources = []
    for chunk in result.get("retrieved_chunks", []):
        formatted_sources.append({
            "filename": chunk.get("source", "Unknown"),
            "page": chunk.get("page", 1),
            "score": round(chunk.get("rrf_score", 0.0), 3),
            "preview": chunk.get("chunk_text", "")[:150],
        })

    return {
        "query": result["query"],
        "answer": result["answer"],
        "faithfulness_score": result["faithfulness_score"],
        "verdict": result["verdict"],
        "sentence_scores": result["sentence_scores"],
        "sources": formatted_sources,
    }


@app.get("/history")
def get_history():
    """Reads the logging CSV and returns the 5 most recent queries."""
    log_file = os.path.join(BASE_DIR, "rag_results_log.csv")
    if not os.path.exists(log_file):
        return []

    try:
        history = []
        with open(log_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            recent = rows[-5:]
            for row in recent:
                history.append({
                    "query": row["query"],
                    "faithfulness_score": float(row["faithfulness_score"]),
                    "verdict": row["verdict"],
                })
        return history[::-1]
    except Exception as e:
        print(f"Error reading history: {e}")
        return []


@app.post("/reset")
def reset_session():
    """Clears all state — no document, no index."""
    global retriever, active_filename, active_chunk_count, active_page_count
    retriever = None
    active_filename = None
    active_chunk_count = 0
    active_page_count = 0
    _clear_upload_dir()
    return {"status": "reset"}


# ── Entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
