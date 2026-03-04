# embedder.py

import os
from uuid import uuid4
from dotenv import load_dotenv

from langchain_qdrant import QdrantVectorStore
from langchain_voyageai import VoyageAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from chunker import chunk_novel

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

# voyage-3-lite outputs 512-dimensional vectors
EMBEDDING_DIM = 512


def get_embeddings() -> VoyageAIEmbeddings:
    return VoyageAIEmbeddings(
        voyage_api_key=VOYAGE_API_KEY,
        model="voyage-3-lite"
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
) -> QdrantVectorStore:
    """
    Embeds a novel PDF into Qdrant Cloud using Voyage AI embeddings,
    or loads an existing collection if already embedded.

    Args:
        pdf_path:        Path to the PDF file. Required only for first-time embedding.
        collection_name: Qdrant collection name (usually the novel's filename stem).

    Returns:
        A QdrantVectorStore instance ready for similarity search.
    """
    if collection_name is None:
        raise ValueError("collection_name must be provided.")

    # Sanitize: lowercase + hyphens only
    collection_name = collection_name.lower().replace("_", "-").replace(" ", "-")

    client = _get_qdrant_client()
    embeddings = get_embeddings()

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

    print(f"✅ {len(chunks)} chunks generated. Uploading in batches...")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )

    # Process in batches to keep memory flat
    BATCH_SIZE = 50
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_ids = [str(uuid4()) for _ in batch]
        vector_store.add_documents(batch, ids=batch_ids)
        print(f"  📤 Uploaded chunks {i+1}–{min(i + BATCH_SIZE, len(chunks))} of {len(chunks)}")

    print(f"✅ All embeddings uploaded to '{collection_name}' successfully.")
    return vector_store