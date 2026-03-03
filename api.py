# api.py

import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from embedder import embed_novel
from multi_query import ask_question

app = FastAPI(
    title="NarrativeAI RAG API",
    description="FastAPI backend for Hybrid RAG (Multi-query + BM25 + Qdrant)",
    version="2.0.0"
)

UPLOAD_TEMP_DIR = "data"
Path(UPLOAD_TEMP_DIR).mkdir(exist_ok=True)


# ----------------------
# UPLOAD & EMBED
# ----------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF upload, embeds it into Qdrant Cloud, and returns
    the collection name to use for querying.
    """
    novel_name = Path(file.filename).stem
    # Sanitize to match Qdrant collection naming rules
    collection_name = novel_name.lower().replace("_", "-").replace(" ", "-")

    pdf_path = os.path.join(UPLOAD_TEMP_DIR, file.filename)

    try:
        # Save uploaded file temporarily
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # embed_novel handles the "already embedded" check internally
        embed_novel(pdf_path=pdf_path, collection_name=collection_name)

    finally:
        # Always clean up the temp PDF after embedding
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

    return {
        "status": "ready",
        "novel": novel_name,
        "collection_name": collection_name
    }


# ----------------------
# QUERY
# ----------------------
class QueryRequest(BaseModel):
    question: str
    collection_name: str


@app.post("/query")
def query_rag(req: QueryRequest):
    """
    Runs the full RAG pipeline for a given question and Qdrant collection.
    """
    result = ask_question(
        question=req.question,
        collection_name=req.collection_name
    )
    return result