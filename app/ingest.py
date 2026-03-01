# app/ingest.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.vector_store import get_vectorstore

PDF_PATH = "data/sample.pdf"   # ← change this to your actual PDF
CHROMA_PATH = "chroma_db"

def ingest():

    # 1️⃣ Load PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    # 2️⃣ Split documents (IMPORTANT: preserves metadata)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    # 3️⃣ Store in Chroma
    vectordb = get_vectorstore()

    vectordb.add_documents(chunks)
   

    print("Ingestion complete ✅")


if __name__ == "__main__":
    ingest()