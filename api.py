# api.py

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import os
from pathlib import Path

from embedder import embed_novel
from multi_query import ask_question

app = FastAPI(
    title="NarrativeAI RAG API",
    description="FastAPI backend for Hybrid RAG (Multi-query + BM25)",
    version="1.0.0"
)

# =====================================================
# CONFIG
# =====================================================
VECTOR_STORE_ROOT = "chroma_db"
Path(VECTOR_STORE_ROOT).mkdir(exist_ok=True)

# =====================================================
# UPLOAD + EMBED (OFFLINE STEP)
# =====================================================
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF and create embeddings (runs once per document).
    Skips embedding if already exists.
    """

    novel_name = Path(file.filename).stem
    persist_dir = os.path.join(VECTOR_STORE_ROOT, novel_name)

    # If embeddings already exist, skip embedding
    if os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0:
        return {
            "status": "already embedded",
            "novel": novel_name,
            "persist_dir": persist_dir
        }

    # Ensure temp folder exists
    os.makedirs("data", exist_ok=True)
    pdf_path = f"data/{file.filename}"

    # Save uploaded PDF temporarily
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Embed + persist
    embed_novel(
        pdf_path=pdf_path,
        persist_dir=persist_dir
    )

    # Optional: delete temp PDF to save space
    os.remove(pdf_path)

    return {
        "status": "embedded",
        "novel": novel_name,
        "persist_dir": persist_dir
    }

# =====================================================
# QUERY (ONLINE STEP)
# =====================================================
class QueryRequest(BaseModel):
    question: str
    persist_dir: str

@app.post("/query")
def query_rag(req: QueryRequest):
    """
    Ask a question over an embedded novel.
    """

    result = ask_question(
        question=req.question,
        persist_dir=req.persist_dir
    )

    return result
