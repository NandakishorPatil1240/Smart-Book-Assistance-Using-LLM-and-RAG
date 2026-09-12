import os
import gc
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def get_embeddings():
    # 429 Rate Limit error saṭhi max_retries=5 add kela ahe
    return MistralAIEmbeddings(
        model="mistral-embed",
        max_retries=5
    )

def build_vector_db(pdf_path, db_path="chroma_db"):
    """
    PDF read karun text chunks banvto ani Chroma Vector Database create karto.
    """
    print(f"Reading PDF: {pdf_path}...")
    
    # 1. Lightweight PDF Extraction (pypdf - No Deprecation Warning)
    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            raw_text += extracted + "\n"

    # 2. Text Chunking (RAM Save Karnyasathi chunk_size=500)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(raw_text)

    # Memory Cleanup
    del raw_text
    gc.collect()

    print(f"Total chunks created: {len(chunks)}. Storing in Vector DB...")

    # 3. Create Chroma Vector Store
    embeddings = get_embeddings()
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )

    del chunks
    gc.collect()
    
    print("Vector database successfully created!")
    return vectorstore

def load_vector_db(db_path="chroma_db"):
    """
    Existing Chroma Vector DB load karnyasaṭhi helper function.
    """
    if os.path.exists(db_path):
        embeddings = get_embeddings()
        return Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
    return None
