import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
import gc

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from create_database import build_vector_db

# Example call:
# build_vector_db("my_book.pdf")

load_dotenv()

st.set_page_config(page_title="RAG Book Assistant")

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document")

# Cached Embedding Object (Mule repeated initialization cha RAM vaachel)
@st.cache_resource
def get_embeddings():
    return MistralAIEmbeddings(model="mistral-embed")

embeddings = get_embeddings()

uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):
        with st.spinner("Processing document efficiently..."):
            # 1. Lightweight PDF Text Extraction
            reader = PdfReader(file_path)
            raw_text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    raw_text += extracted + "\n"

            # Temp file delete kara
            if os.path.exists(file_path):
                os.remove(file_path)

            # 2. Text Chunking
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,       # Chunk size 1000 varun 500 keli RAM bachavnyasathi
                chunk_overlap=50
            )
            chunks = splitter.split_text(raw_text)

            # Clean raw text from memory
            del raw_text
            gc.collect()

            # 3. Create Chroma Vectorstore
            vectorstore = Chroma.from_texts(
                texts=chunks,
                embedding=embeddings,
                persist_directory="chroma_db"
            )

            del chunks
            gc.collect()

        st.success("Vector database created successfully!")

# Ask Question Logic
if os.path.exists("chroma_db"):
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity", # MMR multi-search algorithm peksha similarity light aste
        search_kwargs={"k": 3}
    )

    llm = ChatMistralAI(model="mistral-small-latest")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.
Use ONLY the provided context to answer the question.
If the answer is not present in the context, say: "I could not find the answer in the document."
"""
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion:\n{question}"
            )
        ]
    )

    st.divider()
    st.subheader("Ask Questions From the Book")

    query = st.text_input("Enter your question")

    if query:
        with st.spinner("Searching and generating response..."):
            docs = retriever.invoke(query)

            context = "\n\n".join([doc.page_content for doc in docs])

            final_prompt = prompt.invoke({
                "context": context,
                "question": query
            })

            response = llm.invoke(final_prompt)

            st.write("### AI Answer")
            st.write(response.content)
