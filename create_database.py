import os
import gc
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

# Embedding Model with Retry Logic (Fixes 429 Error)
def get_embeddings():
    return MistralAIEmbeddings(
        model="mistral-embed",
        max_retries=5  # 429 Rate Limit error pasun vachavnyasathi
    )

# PDF Processing and Vector Store Creation
def create_vector_db(file_path, db_path="chroma_db"):
    # 1. Read PDF Text (pypdf - Lightweight)
    reader = PdfReader(file_path)
    raw_text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            raw_text += extracted + "\n"

    # 2. Text Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(raw_text)

    # Free raw text memory
    del raw_text
    gc.collect()

    # 3. Save to Chroma VectorDB
    embeddings = get_embeddings()
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )

    del chunks
    gc.collect()
    return vectorstore

# Load existing Vector DB
def load_vector_db(db_path="chroma_db"):
    if os.path.exists(db_path):
        embeddings = get_embeddings()
        return Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
    return None
