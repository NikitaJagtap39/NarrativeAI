# 📚 NarrativeAI

An AI-powered Q&A system for novels and storybooks, built with a Hybrid RAG (Retrieval-Augmented Generation) architecture. Upload any book as a PDF and ask questions about it — NarrativeAI retrieves the most relevant passages and generates accurate, grounded answers.

## Features

- **Hybrid RAG** — combines semantic vector search with BM25 keyword reranking for higher retrieval precision
- **Multi-query retrieval** — generates multiple rephrasings of each question to improve recall
- **Anti-hallucination prompting** — answers are strictly grounded in retrieved passages; the model will not infer or guess
- **Smart PDF ingestion** — uses an LLM to detect where the story begins and ends, stripping noise like acknowledgements, copyright pages, and promotional content
- **Interactive UI** — Streamlit frontend with retrieval transparency, showing vector and BM25 scores for each retrieved passage

## Architecture

### 1. Document Loading & Cleaning
- PDFs are loaded using **PyMuPDF**
- A Gemini LLM analyzes the first and last pages to detect story boundaries, removing front/back matter automatically

### 2. Chunking & Embedding
- Text is split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter` (1000 tokens, 150 overlap)
- Chunks are embedded using **Voyage AI** (`voyage-4-large`) and stored in **Qdrant Cloud**
- Embeddings are uploaded in batches to keep memory usage flat during ingestion

### 3. Multi-Query Retrieval
- Each user question is rephrased into 5 variants by a Gemini LLM
- All variants are run against Qdrant's vector store, retrieving the top 20 documents per query
- Results are deduplicated and filtered by a cosine similarity threshold to remove weak matches

### 4. BM25 Reranking (Reversed Hybrid RAG)
- The filtered semantic results are reranked using **BM25** keyword scoring
- This "semantic first, lexical second" approach forms the Reversed Hybrid RAG pattern — capturing broad relevance via embeddings, then sharpening precision via keyword matching

### 5. Answer Generation
- The top 5 reranked passages are passed to a Gemini LLM with a strict grounding prompt
- If the answer isn't in the retrieved context, the model says so rather than hallucinating

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | Voyage AI `voyage-4-large` |
| Vector DB | Qdrant Cloud |
| LLM | Google Gemini (`gemini-2.5-flash-lite`) |
| RAG Framework | LangChain |
| PDF Parsing | PyMuPDF |
| BM25 Reranking | `rank-bm25` |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Deployment | Render (Docker) |

## Running Locally

### Prerequisites
- Docker
- A Qdrant Cloud account ([cloud.qdrant.io](https://cloud.qdrant.io))
- A Voyage AI API key ([dashboard.voyageai.com](https://dashboard.voyageai.com))
- A Google AI API key ([aistudio.google.com](https://aistudio.google.com))

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/narrativeai
cd narrativeai
```

2. Create a `.env` file:
```env
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
VOYAGE_API_KEY=your_voyage_api_key
GOOGLE_API_KEY=your_google_api_key
```

3. Build and run with Docker:
```bash
docker build -t narrativeai .
docker run --env-file .env -p 8000:8000 -p 8001:8001 narrativeai
```

4. Open [http://localhost:8000](http://localhost:8000) in your browser.

## Usage

1. Upload a novel or storybook as a PDF using the file uploader
2. Wait for the book to be processed and embedded (this takes ~40–50 seconds for a full novel)
3. Select the book from the sidebar library
4. Ask any question about the book and receive a grounded answer with source passages
