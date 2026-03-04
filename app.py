# app.py

import streamlit as st
import requests
from pathlib import Path

API_URL = "http://127.0.0.1:8001"

st.set_page_config(page_title="Narrative AI", layout="wide")
st.title("📚 Narrative AI")

# ===========================
# SESSION STATE
# ===========================
if "novels" not in st.session_state:
    # Each entry: {"display_name": str, "collection_name": str}
    st.session_state["novels"] = []

# ===========================
# UPLOAD
# ===========================
uploaded_file = st.file_uploader("Upload a book (PDF)", type=["pdf"])

if uploaded_file:
    already_uploaded = any(
        n["display_name"] == Path(uploaded_file.name).stem
        for n in st.session_state["novels"]
    )

    if already_uploaded:
        st.info("This novel is already in your library.")
    else:
        with st.status("Processing PDF... this may take a minute ⏳"):
            r = requests.post(f"{API_URL}/upload", files={"file": uploaded_file})

            if r.status_code == 200:
                data = r.json()
                st.session_state["novels"].append({
                    "display_name": data["novel"],
                    "collection_name": data["collection_name"]
                })
                st.success(f"✅ '{data['novel']}' is ready for Q&A!")
            else:
                st.error(f"Upload failed: {r.text}")

# ===========================
# SIDEBAR — LIBRARY
# ===========================
st.sidebar.header("📖 Library")

if not st.session_state["novels"]:
    st.sidebar.info("No novels yet. Upload a PDF to get started.")
    selected_novel = None
else:
    novel_names = [n["display_name"] for n in st.session_state["novels"]]
    selected_display = st.sidebar.selectbox("Choose Novel:", novel_names)
    selected_novel = next(
        (n for n in st.session_state["novels"] if n["display_name"] == selected_display),
        None
    )

# ===========================
# Q&A SECTION
# ===========================
if selected_novel:
    st.header(f"💬 Q&A: {selected_novel['display_name']}")
    user_question = st.text_input("Ask a question about the book:")

    if st.button("Get Answer") and user_question:
        with st.spinner("Searching and generating answer..."):
            payload = {
                "question": user_question,
                "collection_name": selected_novel["collection_name"]
            }
            r = requests.post(f"{API_URL}/query", json=payload)

            if r.status_code == 200:
                data = r.json()

                # ---- Answer ----
                st.subheader("🤖 Answer")
                st.write(data["answer"])

                # ---- Retrieval Details ----
                with st.expander("🔍 Retrieval Details"):
                    st.markdown("**Generated Query Variants:**")
                    for i, q in enumerate(data.get("generated_queries", []), 1):
                        st.markdown(f"{i}. {q}")

                    st.markdown("---")
                    st.markdown("**Top 5 Retrieved Passages:**")

                    docs = data.get("retrieved_docs", [])
                    if not docs:
                        st.warning("No relevant passages found above the relevance threshold.")
                    else:
                        for i, doc in enumerate(docs, 1):
                            st.markdown(f"**Passage {i}**")
                            col1, col2 = st.columns(2)

                            with col1:
                                vec_score = doc.get("vector_score")
                                label = f"{vec_score:.4f}" if vec_score is not None else "N/A"
                                st.markdown(
                                    f"<div style='background:#d4edda;padding:6px;border-radius:5px;"
                                    f"text-align:center;font-size:0.85em;'>"
                                    f"🔵 Vector Score: <b>{label}</b></div>",
                                    unsafe_allow_html=True
                                )
                            with col2:
                                bm25_score = doc.get("bm25_score")
                                label = f"{bm25_score:.4f}" if bm25_score is not None else "N/A"
                                st.markdown(
                                    f"<div style='background:#d4edda;padding:6px;border-radius:5px;"
                                    f"text-align:center;font-size:0.85em;'>"
                                    f"🟠 BM25 Score: <b>{label}</b></div>",
                                    unsafe_allow_html=True
                                )

                            st.write(doc["content"])
                            st.markdown("---")
            else:
                st.error(f"Query failed: {r.text}")