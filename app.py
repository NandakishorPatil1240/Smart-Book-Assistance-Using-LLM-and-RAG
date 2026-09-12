import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
import gc

# LightweightImports (No Deprecation Warnings)
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma  # Direct lightweight package
from langchain_core.prompts import ChatPromptTemplate
from create_database import build_vector_db, get_embeddings

load_dotenv()

st.set_page_config(page_title="RAG Book Assistant")

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document")

# Cached Embedding Object saṭhi function
@st.cache_resource
def load_embeddings():
    return get_embeddings()

embeddings = load_embeddings()

uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")

if uploaded_file:
    # Uploaded PDF file temporarily save kara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):
        with st.spinner("Processing document efficiently..."):
            # create_database.py madhil function call
            build_vector_db(file_path)

            # Temp file delete kara
            if os.path.exists(file_path):
                os.remove(file_path)

            gc.collect()

        st.success("Vector database created successfully!")

# Question Answering Section
if os.path.exists("chroma_db"):
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    # Top 2 Chunks + Fast Similarity Search (429 Rate Limit Avoid saṭhi)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

    # max_retries=5 mul 429 Error ala tar auto-retry hoil
    llm = ChatMistralAI(
        model="mistral-small-latest",
        max_retries=5
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
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

            # Memory Cleanup
            del docs
            del context
            del final_prompt
            del response
            gc.collect()
