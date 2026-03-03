# multi_query.py

import os
from typing import List, Tuple
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from embedder import embed_novel, get_embeddings, _get_qdrant_client, _collection_has_vectors

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# =====================================================
# LLMs
# =====================================================
# Both query generation and answer generation use Gemini
# so no OpenAI quota is needed
query_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    api_key=GOOGLE_API_KEY
)

answer_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    api_key=GOOGLE_API_KEY
)

# =====================================================
# MULTI-QUERY PROMPT & PARSER
# =====================================================
multi_query_gen_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant helping retrieve passages from a novel.
Generate 5 different rephrasings of the user's question to improve document retrieval.
Each rephrasing should be on its own line. Do not number them or add bullet points.

Original question: {question}
""")


class LineListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        return [line.strip() for line in text.strip().split("\n") if line.strip()]


query_parser = LineListOutputParser()
multi_query_gen_chain = multi_query_gen_prompt | query_llm | query_parser

# =====================================================
# ANTI-HALLUCINATION ANSWER PROMPT
# =====================================================
answer_prompt = ChatPromptTemplate.from_template("""
You are an assistant that answers questions strictly based on the provided excerpts from a novel.

Rules you must follow:
- ONLY use information explicitly stated in the context below.
- If the answer is not found in the context, say exactly: "I couldn't find that information in the provided text."
- Do NOT infer, assume, or use any outside knowledge about the story or its characters.
- Answer in clear, detailed paragraph form. Reference the text directly when possible.
- Do NOT use bullet points or numbered lists.

Context:
{context}

Question: {question}

Answer:
""")

# =====================================================
# RELEVANCE FILTER
# =====================================================
# Qdrant with cosine metric returns similarity scores between 0 and 1.
# Higher = more similar. We discard docs below this threshold to avoid
# feeding weak/irrelevant context to the LLM — a primary cause of hallucinations.
SIMILARITY_THRESHOLD = 0.4  # Tune between 0.4–0.7 based on your results


def filter_by_relevance(
    docs_with_scores: List[Tuple[Document, float]],
    threshold: float = SIMILARITY_THRESHOLD
) -> List[Document]:
    """
    Filters out low-relevance documents below the cosine similarity threshold.
    Returns an empty list if nothing passes — the caller handles this gracefully
    rather than sending low-quality context to the LLM.
    """
    filtered = [doc for doc, score in docs_with_scores if score >= threshold]
    rejected = len(docs_with_scores) - len(filtered)

    if not filtered:
        print(f"⚠️  No documents passed the relevance threshold of {threshold}.")
    else:
        print(f"✅ {len(filtered)} passed / {rejected} rejected by relevance filter (threshold={threshold}).")

    return filtered


# =====================================================
# MULTI-QUERY RETRIEVAL
# =====================================================
def get_multi_query_docs(
    user_question: str,
    collection_name: str,
    pdf_path: str = None,
    top_k: int = 20
) -> Tuple[List[Document], List[str]]:
    """
    Generates multiple query variants, retrieves docs from Qdrant for each,
    applies relevance filtering, and returns deduplicated results.

    Returns:
        (filtered_docs, generated_queries)
    """
    collection_name = collection_name.lower().replace("_", "-").replace(" ", "-")
    client = _get_qdrant_client()
    embeddings = get_embeddings()

    # Embed novel if not already done
    if not _collection_has_vectors(client, collection_name):
        if pdf_path is None:
            # Upload step failed or didn't complete — return gracefully so UI can show a message
            return [], ["Upload may have failed — please re-upload the PDF and try again."]
        embed_novel(pdf_path=pdf_path, collection_name=collection_name)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )

    # Generate query variants
    generated_queries = multi_query_gen_chain.invoke({"question": user_question})
    print(f"🔍 Generated {len(generated_queries)} query variants.")

    # Retrieve docs for each query variant with scores
    all_docs_with_scores: List[Tuple[Document, float]] = []
    for q in generated_queries:
        results = vector_store.similarity_search_with_score(q, k=top_k)
        for doc, score in results:
            doc.metadata["vector_score"] = float(score)
        all_docs_with_scores.extend(results)

    # Deduplicate by content (keep highest score per unique chunk)
    seen: dict = {}
    for doc, score in all_docs_with_scores:
        key = doc.page_content[:200]
        if key not in seen or score > seen[key][1]:
            seen[key] = (doc, score)

    deduped = list(seen.values())
    print(f"📄 {len(deduped)} unique chunks retrieved before relevance filtering.")

    # Apply relevance filter
    filtered_docs = filter_by_relevance(deduped, threshold=SIMILARITY_THRESHOLD)

    return filtered_docs, generated_queries


# =====================================================
# BM25 RERANKING
# =====================================================
def bm25_rerank(query: str, docs: List[Document]) -> List[Document]:
    """
    Reranks documents by BM25 keyword score. This forms the 'Reversed Hybrid RAG'
    approach: semantic retrieval first, then lexical reranking.
    """
    if not docs:
        return []

    corpus = [doc.page_content for doc in docs]
    tokenized_corpus = [text.split() for text in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    scores = bm25.get_scores(query.split())
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    ranked_docs = []
    for score, doc in ranked:
        doc.metadata["bm25_score"] = float(score)
        ranked_docs.append(doc)

    return ranked_docs


# =====================================================
# ANSWER GENERATION
# =====================================================
def generate_answer(question: str, docs: List[Document]) -> str:
    """
    Generates an answer from the top docs. If no relevant docs exist,
    returns a safe fallback message instead of hallucinating.
    """
    if not docs:
        return "I couldn't find relevant information in the book to answer this question."

    context = "\n\n".join(doc.page_content for doc in docs[:5])
    chain = answer_prompt | answer_llm
    response = chain.invoke({"context": context, "question": question})
    return response.content


# =====================================================
# MAIN ENTRY POINT (called by api.py)
# =====================================================
def ask_question(question: str, collection_name: str) -> dict:
    """
    Full RAG pipeline:
      1. Multi-query retrieval from Qdrant
      2. Relevance filtering
      3. BM25 reranking
      4. Answer generation with grounded prompt

    Returns a dict with answer, generated_queries, and retrieved_docs.
    """
    # Step 1: Retrieve + filter
    docs, generated_queries = get_multi_query_docs(
        user_question=question,
        collection_name=collection_name,
        top_k=20
    )

    # Step 2: BM25 rerank
    reranked = bm25_rerank(question, docs)

    # Step 3: Generate answer from top 5
    answer = generate_answer(question, reranked[:5])

    # Step 4: Format output for UI
    retrieved_docs = [
        {
            "content": d.page_content,
            "vector_score": d.metadata.get("vector_score"),
            "bm25_score": d.metadata.get("bm25_score"),
            "source": d.metadata.get("source", None)
        }
        for d in reranked[:5]
    ]

    return {
        "answer": answer,
        "generated_queries": generated_queries,
        "retrieved_docs": retrieved_docs
    }