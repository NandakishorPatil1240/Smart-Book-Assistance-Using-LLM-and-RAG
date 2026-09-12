import os
import gc
from dotenv import load_dotenv

# Lightweight packages (DeprecationWarning मुक्त)
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Initialize Embeddings (max_retries add kela ahe 429 Error saṭhi)
embedding_model = MistralAIEmbeddings(
    model="mistral-embed",
    max_retries=5
)

# 2. Load Vectorstore (ChromaDB)
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# 3. Memory-friendly Retriever (k=2 karun API load kammi kela)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# 4. LLM Setup (max_retries=5 sobat)
llm = ChatMistralAI(
    model="mistral-small-latest",
    max_retries=5
)

# 5. Prompt Template
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

print("RAG CLI System Ready!")
print("Press 0 to exit")

# 6. Optimized Interactive Loop
while True:
    try:
        query = input("\nYou: ").strip()

        if query == "0":
            print("Exiting...")
            break

        if not query:
            continue

        # Retrieval & Prompt Invocation
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        response = llm.invoke(final_prompt)

        print(f"\nAI: {response.content}")

        # Explicit Memory Cleanup
        del docs
        del context
        del final_prompt
        del response
        gc.collect()

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
