"""
ingest.py – Load and index documents into a vector store (FAISS)
"""

import os
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

DATA_DIR = "data/your_documents"
DB_DIR = "data/vector_store"

def load_documents():
    documents = []
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        else:
            loader = TextLoader(filepath)
        documents.extend(loader.load())
    return documents

def ingest_documents():
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(DB_DIR)
    print(f"Ingested and saved {len(docs)} chunks to FAISS at {DB_DIR}")

if __name__ == "__main__":
    ingest_documents()
