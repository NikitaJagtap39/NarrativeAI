# multi_query.py

from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import BaseOutputParser
from typing import List
import os
from dotenv import load_dotenv

from embedder import embed_novel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- API Keys from Streamlit secrets ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- LLMs ---
query_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
answer_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    api_key=GOOGLE_API_KEY
)

# --- Multi-query prompt ---
multi_query_gen_prompt_template = """
You are an AI assistant. Generate 5 different rephrasings of the user's question
to improve document retrieval from a vector database.
Each question should be on a new line.

Original question: {question}
"""
multi_query_gen_prompt = ChatPromptTemplate.from_template(
    multi_query_gen_prompt_template
)

# --- Output parser ---
class LineListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return [line.strip() for line in lines if line.strip()]

query_parser = LineListOutputParser()

# --- Multi-query chain ---
multi_query_gen_chain = multi_query_gen_prompt | query_llm | query_parser

# --- Answer prompt ---
answer_prompt = ChatPromptTemplate.from_template("""
Use ONLY the context below.
Give a detailed, explanatory answer in paragraph form.
Include specific examples or descriptions from the text when possible.
Do NOT summarize in bullet points.
If the context is insufficient, say so clearly.

Context:
{context}

Question:{question}

Answer:
""")

# =====================================================
# MULTI-QUERY RETRIEVAL WITH VECTOR SCORES
# =====================================================
def get_multi_query_docs(user_question: str, persist_dir: str, pdf_path: str = None):
    """
    Returns documents with VECTOR similarity scores + generated queries.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Load or embed
    if not os.path.exists(persist_dir) or len(os.listdir(persist_dir)) == 0:
        if pdf_path is None:
            raise ValueError("No embeddings found. PDF required.")
        vector_store = embed_novel(pdf_path=pdf_path, persist_dir=persist_dir)
    else:
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )

    # ---- Multi-query generation ----
    generated_queries = multi_query_gen_chain.invoke(
        {"question": user_question}
    )

    all_docs = []

    # ---- Retrieve documents WITH similarity score ----
    for q in generated_queries:
        results = vector_store.similarity_search_with_score(q, k=20)

        for doc, score in results:
            # store vector similarity score
            doc.metadata["vector_score"] = float(score)
            all_docs.append(doc)

    # Deduplicate documents
    unique_docs = {}
    for doc in all_docs:
        key = doc.page_content[:200]
        if key not in unique_docs:
            unique_docs[key] = doc

    return list(unique_docs.values()), generated_queries


# =====================================================
# ANSWER GENERATION
# =====================================================
def generate_answer(question: str, docs):
    context = "\n\n".join(doc.page_content for doc in docs[:5])
    chain = answer_prompt | answer_llm
    response = chain.invoke(
        {"context": context, "question": question}
    )
    return response.content


# =====================================================
# BM25 RERANKING WITH SCORES
# =====================================================
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

def bm25_rerank(query: str, docs: List[Document]) -> List[Document]:
    """
    Rerank documents using BM25 and attach BM25 scores.
    """

    corpus = [doc.page_content for doc in docs]
    tokenized_corpus = [text.split() for text in corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    ranked_docs = []
    for score, doc in ranked:
        doc.metadata["bm25_score"] = float(score)
        ranked_docs.append(doc)

    return ranked_docs

# =====================================================
# FASTAPI ENTRY FUNCTION
# =====================================================
def ask_question(question: str, persist_dir: str):

    docs, generated_queries = get_multi_query_docs(
        user_question=question,
        persist_dir=persist_dir
    )

    reranked = bm25_rerank(question, docs)
    answer = generate_answer(question, reranked)

    return {
        "answer": answer,
        "generated_queries": generated_queries,
        "sources": [
            {
                "vector_score": d.metadata.get("vector_score"),
                "bm25_score": d.metadata.get("bm25_score")
            }
            for d in reranked[:5]
        ]
    }