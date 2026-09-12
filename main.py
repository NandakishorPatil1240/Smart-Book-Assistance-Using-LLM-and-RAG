import os
import gc
from dotenv import load_dotenv

# Lightweight packages
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma  # Community import aevaji direct langchain_chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Initialize Embeddings
embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

# 2. Load Vectorstore (ChromaDB)
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# 3. Memory-friendly Similarity Retriever (MMR aevaji similarity)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Top 3 chunks kafi ahet RAM save karnyasathi
)

# 4. Initialize LLM Model
llm = ChatMistralAI(model="mistral-small-latest")

# 5. Prompt Template Setup
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
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("RAG system created successfully!")
print("Press 0 to exit")

# 6. Optimized Loop
while True:
    try:
        query = input("\nYou: ").strip()

        if query == "0":
            print("Exiting...")
            break

        if not query:
            continue

        # Document Retrieval
        docs = retriever.invoke(query)

        context = "\n\n".join([doc.page_content for doc in docs])

        # Prompt Creation & LLM Call
        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        response = llm.invoke(final_prompt)

        print(f"\nAI: {response.content}")

        # Explicit Memory Cleanup inside continuous loop
        del docs
        del context
        del final_prompt
        del response
        gc.collect()

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
