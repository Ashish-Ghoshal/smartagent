"""
retriever.py – Load the FAISS vector store and return a retriever
"""

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

DB_DIR = "data/vector_store"

def get_retriever():
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever()
