from loader import extract_clean_story_pages
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_novel(pdf_path: str, chunk_size: int = 1000, overlap: int =150):
    """
    Extracts and cleans the pages from a PDF, then chunks them using RecursiveCharacterSplitter.
    Works directly with Document objects returned by extract_clean_story_pages.

    Returns a list of chunked Document objects.
    """

    #Importing function to retrieve cleaned list of document objects
    docs = extract_clean_story_pages(pdf_path)

    #Initialize the text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        add_start_index=True
    )

    #Split the document objects into chunks
    chunks = splitter.split_documents(docs)

    return chunks