"""
FastAPI Backend for Smart RAG
Serves endpoints for queries, status, and history.
"""

import os
# Prevent huggingface tokenizers threading deadlock on Windows FastAPI!
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import pipeline elements
from main import build_pipeline, run_rag

app = FastAPI(title="Smart RAG API", description="Backend for the Smart RAG dashboard")

# CORS middleware to allow the frontend to access these endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold the RAG pipeline components
retriever = None

@app.on_event("startup")
def startup_event():
    """Build the RAG pipeline into memory when the server starts."""
    global retriever
    print("\n--- Initializing Smart RAG Backend ---")
    retriever = build_pipeline()
    print("--- Backend Initialization Complete ---\n")


class QueryRequest(BaseModel):
    query: str


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Returns the index.html file from the /static folder."""
    static_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_file):
        with open(static_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Smart RAG API</h1><p>index.html not found in /static folder</p>", status_code=404)


@app.get("/status")
def get_status():
    """Returns system status and basic metric counts."""
    raw_folder = os.path.join(os.path.dirname(__file__), "data", "raw")
    pdf_count = 0
    if os.path.exists(raw_folder):
        pdf_count = len([f for f in os.listdir(raw_folder) if f.endswith(".pdf")])
        
    return {
        "status": "online",
        "documents": pdf_count,
        "chunks": 0, # Placeholder
        "model": "GPT-4o-mini",
        "retrieval": "Hybrid RRF"
    }


@app.post("/query")
def handle_query(request: QueryRequest):
    """
    Core RAG endpoint. Runs the query, evaluates hallucination,
    and returns formatted structured data.
    """
    global retriever
    if not retriever:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")

    # run_rag natively runs semantic chunk retrieval, LLM generation, 
    # and NLI hallucination detector (score_faithfulness) sequentially!
    result = run_rag(request.query, retriever)

    # Re-structure the output specifically for the frontend UI
    formatted_sources = []
    for chunk in result.get("retrieved_chunks", []):
        formatted_sources.append({
            "filename": chunk.get("source", "Unknown"),
            "page": chunk.get("page", 1),
            "score": round(chunk.get("rrf_score", 0.0), 3),
            "preview": chunk.get("chunk_text", "")[:150]
        })

    return {
        "query": result["query"],
        "answer": result["answer"],
        "faithfulness_score": result["faithfulness_score"],
        "verdict": result["verdict"],
        "sentence_scores": result["sentence_scores"],
        "sources": formatted_sources
    }


@app.get("/history")
def get_history():
    """Reads the logging CSV and returns the 5 most recent queries."""
    log_file = os.path.join(os.path.dirname(__file__), "rag_results_log.csv")
    if not os.path.exists(log_file):
        return []
        
    try:
        df = pd.read_csv(log_file)
        if df.empty:
            return []
            
        recent = df.tail(5)
        history = []
        for _, row in recent.iterrows():
            history.append({
                "query": row["query"],
                "faithfulness_score": row["faithfulness_score"],
                "verdict": row["verdict"]
            })
            
        # Reverse to return newest logs first
        return history[::-1]
    except Exception as e:
        print(f"Error reading history: {e}")
        return []


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
