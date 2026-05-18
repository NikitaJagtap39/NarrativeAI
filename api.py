# api.py

import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
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


# ✅ Health check endpoint for Docker + EC2
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # ✅ Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    novel_name = Path(file.filename).stem
    collection_name = novel_name.lower().replace("_", "-").replace(" ", "-")
    pdf_path = os.path.join(UPLOAD_TEMP_DIR, file.filename)

    try:
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        embed_novel(pdf_path=pdf_path, collection_name=collection_name)
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

    return {
        "status": "ready",
        "novel": novel_name,
        "collection_name": collection_name
    }


class QueryRequest(BaseModel):
    question: str
    collection_name: str


@app.post("/query")
def query_rag(req: QueryRequest):
    result = ask_question(
        question=req.question,
        collection_name=req.collection_name
    )
    return result