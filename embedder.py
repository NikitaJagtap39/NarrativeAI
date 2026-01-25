from chunker import chunk_novel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from uuid import uuid4
import os


def embed_novel(
    pdf_path: str = None,
    persist_dir: str = None,
    model_name="BAAI/bge-base-en-v1.5",
    device="cpu"
):

    if persist_dir is None:
        raise ValueError("persist_dir must be provided.")

    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    index_dir = os.path.join(persist_dir, "index")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    # ✅ CASE 1: embeddings already exist
    if os.path.exists(index_dir):
        print(f"✅ Loading existing embeddings from {persist_dir}")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )

    # ✅ CASE 2: no embeddings yet
    if pdf_path is None:
        raise ValueError(
            f"No embeddings found in '{persist_dir}'. PDF required for first-time embedding."
        )

    print("📘 Chunking novel...")
    chunks = chunk_novel(pdf_path)

    if not chunks:
        raise ValueError("❌ No chunks generated. Check PDF loader / chunker.")

    print(f"✅ {len(chunks)} chunks generated")

    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    ids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(chunks, ids=ids)



    print("✅ Embeddings created and stored successfully")

    return vector_store
