# **Project Overview:**

### This project aims to answer questions to any novel or story book or your choice using a Retrieval-Augmented Generation (RAG) system. It implements a hybrid RAG approach which is semantic similarity and keyword-based retrieval for more accurate results and uses LangChain for all core pipeline steps, ensuring modularity and scalability, and presents results through a Streamlit UI.

#### The workflow is as follows:-

#### 1. Document Loading and Chunking:
* The system loads documents using DocumentLoader classes.
* Since PDFs are used, PyMuPDF is used via PyMuPDFLoader.
* An LLM is also used to analyze first and last pages to detect where the content begins and ends. This step helps in the removal of noisy, promotional content, such as acknowlegements, copyright notices, etc.
* LangChain handles reading, preprocessing, and structuring text for downstream steps.

#### 2. Text Chunking & Embedding:
* Loaded text is divided into smaller chunks using LangChain's text splitters.
* Each chunk is embedded using Hugging Face BAAI bge-en-v1.5 via LangChain's embedding modules.
* Embeddings are stored in a Chroma vector database for fast similarity searches.

#### 3. Multi-Query Retrieval:
* For each user query, LangChain generates multiple related queries automatically.
* Using vector similarity search, it retrieves the top 20 most relevant documents from Chroma.

#### 4. BM25 Keyword-Based Search (Reversed Hybrid RAG):
* The 20 retrived documents are reranked using BM25, which prioritizes the documents with stronger keyword matches.
* This two-step process of semantic first, then lexical retrieval forms the Reversed Hybrid RAG approach.

#### 5. Answer Generation & Streamlit UI:
* After reranking, the final top 5 documents are passed through LangChain LLM chains for coherent, context-aware answer generation.
* Users see the results directly on the Streamlit interface, enabling full interactive Q&A without leaving the app.


