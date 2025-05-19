"""
document_search.py – Tool to query documents using RAG
"""

from langchain_core.tools import tool
from rag.retriever import get_retriever

retriever = get_retriever()

@tool
def document_search(query: str) -> str:
    """Search company documents for information related to the query."""
    results = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in results[:3]])
