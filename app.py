# app.py

import streamlit as st
from pathlib import Path
import os
import tempfile

from multi_query import get_multi_query_docs, generate_answer, bm25_rerank

VECTOR_STORE_ROOT = "chroma_db"
Path(VECTOR_STORE_ROOT).mkdir(exist_ok=True)

st.set_page_config(page_title="Narrative AI", layout="wide")
st.title("📚 Narrative AI")

# --- Session State Initialization ---
if "novel_list" not in st.session_state:
    st.session_state["novel_list"] = [
        d for d in os.listdir(VECTOR_STORE_ROOT)
        if os.path.isdir(os.path.join(VECTOR_STORE_ROOT, d))
    ]

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a book (PDF)", type=["pdf"])
if uploaded_file:
    novel_name = Path(uploaded_file.name).stem
    persist_dir = os.path.join(VECTOR_STORE_ROOT, novel_name)

    if novel_name not in st.session_state["novel_list"]:
        with st.status("Processing PDF...") as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            from embedder import embed_novel
            embed_novel(tmp_path, persist_dir=persist_dir)
            st.session_state["novel_list"].append(novel_name)
            os.remove(tmp_path)
            status.update(label="✅ Ready!", state="complete")

# --- Sidebar ---
st.sidebar.header("Library")
selected_novel = st.sidebar.selectbox(
    "Choose Novel:",
    st.session_state["novel_list"]
)

# --- QnA Logic ---
if selected_novel:
    st.header(f"QnA: {selected_novel}")
    user_question = st.text_input("Ask about the book:")

    persist_dir = os.path.join(VECTOR_STORE_ROOT, selected_novel)

    if st.button("Get Answer") and user_question:
        with st.spinner("Searching through book..."):
            try:
                # --------------------------------------------------
                # Retrieval
                # --------------------------------------------------
                docs, generated_queries = get_multi_query_docs(
                    user_question=user_question,
                    persist_dir=persist_dir
                )

                top_20 = docs[:20]

                # BM25 reranking
                bm25_docs = bm25_rerank(user_question, top_20)
                top_docs = bm25_docs[:5]

                # --------------------------------------------------
                # Answer
                # --------------------------------------------------
                answer = generate_answer(user_question, top_docs)

                st.divider()
                st.subheader("🤖 Agent Answer")
                st.write(answer)

                # --------------------------------------------------
                # Debug / Observability
                # --------------------------------------------------
                with st.expander("🔍 Retrieval Details"):
                    st.write("### 🧠 Generated Queries")
                    st.write(generated_queries)

                    st.write("### 📄 Top Context Passages (Used for Answer)")

                    for i, doc in enumerate(top_docs, 1):
                        vector_score = doc.metadata.get("vector_score", "N/A")
                        bm25_score = doc.metadata.get("bm25_score", "N/A")

                        st.markdown(f"""
**Passage {i}**

• **Vector similarity score:** `{vector_score}`  
• **BM25 score:** `{bm25_score}`  

{doc.page_content[:600]}…
""")

            except Exception as e:
                st.error(f"Error: {str(e)}")
