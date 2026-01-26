# app.py

import streamlit as st
from pathlib import Path
import os
import requests

API_URL = "http://127.0.0.1:8000"  # FastAPI backend
VECTOR_STORE_ROOT = "chroma_db"
Path(VECTOR_STORE_ROOT).mkdir(exist_ok=True)

st.set_page_config(page_title="Narrative AI", layout="wide")
st.title("📚 Narrative AI")

# ===========================
# SESSION STATE
# ===========================
if "novel_list" not in st.session_state:
    st.session_state["novel_list"] = [
        d for d in os.listdir(VECTOR_STORE_ROOT)
        if os.path.isdir(os.path.join(VECTOR_STORE_ROOT, d))
    ]

# ===========================
# UPLOAD
# ===========================
uploaded_file = st.file_uploader("Upload a book (PDF)", type=["pdf"])

if uploaded_file:
    with st.status("Processing PDF..."):
        files = {"file": uploaded_file}
        r = requests.post(f"{API_URL}/upload", files=files)

        if r.status_code == 200:
            novel_name = Path(uploaded_file.name).stem
            st.session_state["novel_list"].append(novel_name)
            st.success("✅ Ready!")
        else:
            st.error("Upload failed")

# ===========================
# SIDEBAR
# ===========================
st.sidebar.header("Library")
selected_novel = st.sidebar.selectbox(
    "Choose Novel:",
    st.session_state["novel_list"]
)

# ===========================
# QnA
# ===========================
if selected_novel:
    st.header(f"QnA: {selected_novel}")
    user_question = st.text_input("Ask about the book:")

    if st.button("Get Answer") and user_question:
        with st.spinner("Thinking..."):
            payload = {
                "question": user_question,
                "persist_dir": f"{VECTOR_STORE_ROOT}/{selected_novel}"
            }

            r = requests.post(f"{API_URL}/query", json=payload)

            if r.status_code == 200:
                data = r.json()
                
                # ✅ Final Answer
                st.subheader("🤖 Agent Answer")
                st.write(data["answer"])

                # ✅ Retrieval Details
                with st.expander("🔍 Retrieval Details"):
                    st.write("Generated Queries:")
                    st.write(data["generated_queries"])
                    
                    st.write("Top 5 Retrieved Documents with Scores:")

                    for doc in data.get("retrieved_docs", [])[:5]:
                        # Display scores inside green boxes
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(
                                f"<div style='background-color:#d4edda;padding:5px;border-radius:5px;text-align:center;'>"
                                f"Vector Score: {doc['vector_score']:.4f}</div>", 
                                unsafe_allow_html=True
                            )
                        with col2:
                            st.markdown(
                                f"<div style='background-color:#d4edda;padding:5px;border-radius:5px;text-align:center;'>"
                                f"BM25 Score: {doc['bm25_score']:.4f}</div>", 
                                unsafe_allow_html=True
                            )
                        st.write(doc["content"])
                        st.write("---")
            else:
                st.error("Query failed")
