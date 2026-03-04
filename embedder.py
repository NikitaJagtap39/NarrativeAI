# embedder.py

import os
from uuid import uuid4
from dotenv import load_dotenv

from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from chunker import chunk_novel

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")         # e.g. https://xyz.qdrant.io:6333
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# BAAI/bge-small-en-v1.5 outputs 384-dimensional vectors
EMBEDDING_DIM = 384


def get_embeddings(model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu") -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )


def _get_qdrant_client() -> QdrantClient:
    """Returns a QdrantClient connected to Qdrant Cloud."""
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError(
            "QDRANT_URL and QDRANT_API_KEY must be set in your .env file. "
            "Get them from https://cloud.qdrant.io"
        )
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def _collection_has_vectors(client: QdrantClient, collection_name: str, retries: int = 3, delay: float = 2.0) -> bool:
    """
    Returns True if the collection exists and already contains points.
    Uses points_count (not vectors_count) for compatibility with newer qdrant-client versions.
    Retries a few times to handle Qdrant's brief indexing delay after upload.
    """
    import time
    for attempt in range(retries):
        try:
            collections = [c.name for c in client.get_collections().collections]
            if collection_name not in collections:
                return False
            info = client.get_collection(collection_name)
            if info.points_count is not None and info.points_count > 0:
                return True
            if attempt < retries - 1:
                print(f"⏳ Waiting for Qdrant to sync points (attempt {attempt + 1}/{retries})...")
                time.sleep(delay)
        except Exception:
            return False
    return False


def _ensure_collection(client: QdrantClient, collection_name: str):
    """Creates the Qdrant collection if it doesn't already exist."""
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        print(f"🆕 Creating Qdrant collection: '{collection_name}'")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        print(f"✅ Collection '{collection_name}' created.")
    else:
        print(f"✅ Collection '{collection_name}' already exists.")


def embed_novel(
    pdf_path: str = None,
    collection_name: str = None,
    model_name: str = "BAAI/bge-small-en-v1.5",
    device: str = "cpu"
) -> QdrantVectorStore:
    """
    Embeds a novel PDF into Qdrant Cloud, or loads an existing collection
    if the novel has already been embedded.

    Args:
        pdf_path:        Path to the PDF file. Required only for first-time embedding.
        collection_name: Qdrant collection name (usually the novel's filename stem).
        model_name:      HuggingFace embedding model to use.
        device:          'cpu' or 'cuda'.

    Returns:
        A QdrantVectorStore instance ready for similarity search.
    """
    if collection_name is None:
        raise ValueError("collection_name must be provided.")

    # Sanitize: lowercase + hyphens only
    collection_name = collection_name.lower().replace("_", "-").replace(" ", "-")

    client = _get_qdrant_client()
    embeddings = get_embeddings(model_name, device)

    _ensure_collection(client, collection_name)

    # If vectors already exist, skip re-embedding
    if _collection_has_vectors(client, collection_name):
        print(f"✅ Embeddings already exist in '{collection_name}'. Skipping re-embedding.")
        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )

    # First-time embedding
    if pdf_path is None:
        raise ValueError(
            f"No vectors found in collection '{collection_name}'. "
            "A PDF path is required for first-time embedding."
        )

    print("📘 Chunking novel...")
    chunks = chunk_novel(pdf_path)

    if not chunks:
        raise ValueError("❌ No chunks generated. Check PDF loader / chunker.")

    print(f"✅ {len(chunks)} chunks generated. Uploading to Qdrant...")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )

    ids = [str(uuid4()) for _ in chunks]
    vector_store.add_documents(chunks, ids=ids)

    print(f"✅ Embeddings uploaded to Qdrant collection '{collection_name}' successfully.")
    return vector_store