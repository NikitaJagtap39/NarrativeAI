import os
from typing import List
from dotenv import load_dotenv

#Extracting page-level text and converts each page to a Document object 
from langchain_community.document_loaders import PyMuPDFLoader 
#Gemini LLM
from langchain_google_genai import ChatGoogleGenerativeAI 
#Structured prompt injection (safe + reusable)
from langchain_core.prompts import PromptTemplate

#Environment Setup
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

#LLM Initialization
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

#Keywords for LLM to look for
NOISE_KEYWORDS = [
    "acknowledgements", "acknowledgments", "all rights reserved",
    "about the author", "dedication", "table of contents",
    "contents", "title page", "publisher", "copyright"
]

PROMO_KEYWORDS = [
    "your next reading list", "discover your next great read",
    "personalized book picks", "also by", "thank you for buying",
    "this is a work of fiction"
]

#Analyzing first few pages to detect where actual information begins
START_PROMPT = PromptTemplate(
    input_variables=["pages", "noise", "promo"],
    template="""
You are analyzing the FIRST pages of a novel.

Noise keywords:
{noise}

Promo keywords:
{promo}

TASK:
Return the 0-based page index where the ACTUAL INFORMATION starts.
- Skip only pages that are purely promotional, copyright notices, table of contents, acknowledgements, or similar marketing content.
- Keep pages that provide context, background, preface, or historical information relevant to understanding the story.

Pages:
{pages}

Return ONLY a single integer.Do not skip context or introductory pages that are informative.
"""
)

#Detecting where the actual story ends
END_PROMPT = PromptTemplate(
    input_variables=["pages", "noise", "promo"],
    template="""
You are analyzing the LAST pages of a novel.

Noise keywords:
{noise}

Promo keywords:
{promo}

TASK:
Return the 0-based page index where the STORY ends.
Ignore previews, author notes, promos.

Pages:
{pages}

Return ONLY a single integer.
"""
)

#Turning multiple document objects into text snippets for the LLM
def format_pages(docs: List, offset: int = 0) -> str:
    blocks = []
    for i, doc in enumerate(docs):
        snippet = doc.page_content[:500].replace("\n", " ") #500 characters are enough for semantic signals
        blocks.append(f"[PAGE {offset + i}]: {snippet}")
    return "\n".join(blocks)    

#Loading page-by-page using PyMuPDF
def extract_clean_story_pages(pdf_path: str) -> List:
    loader = PyMuPDFLoader(pdf_path, mode="page")
    docs = loader.load()

    print(f"\n📄 Total pages extracted: {len(docs)}")

    # ---- LLM boundary detection ----
    start_idx = int(
        llm.invoke(
            START_PROMPT.format(
                pages=format_pages(docs[:10]),
                noise=", ".join(NOISE_KEYWORDS),
                promo=", ".join(PROMO_KEYWORDS),
            )
        ).content.strip()
    )

    end_idx = int(
        llm.invoke(
            END_PROMPT.format(
                pages=format_pages(docs[-10:], offset=len(docs) - 10),
                noise=", ".join(NOISE_KEYWORDS),
                promo=", ".join(PROMO_KEYWORDS),
            )
        ).content.strip()
    )

    print(f"🟢 Relevant START of novel detected at page: {start_idx}")
    print(f"🟢 Relevant END of novel detected at page: {end_idx}")

    # ---- Log skipped front/back pages ----
    if start_idx > 0:
        print(f"❌ Skipped front pages: 0 → {start_idx}")
    
    if end_idx < len(docs) - 1:
        print(f"❌ Skipped back pages: {end_idx + 1} → {len(docs) - 1} ")

    # ---- Middle Pages ---
    story_docs = docs[start_idx:end_idx + 1]

    cleaned = []
    for i, doc in enumerate(story_docs, start=start_idx):
        if not doc.page_content or not doc.page_content.strip():
            print(f"❌ Skipped empty page: {i}")
            continue
        cleaned.append(doc)

    print(f"\n ✅ Pages kept: {len(cleaned)} ")

    return cleaned

