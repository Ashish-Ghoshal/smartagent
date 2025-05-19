"""
summarize_text.py – Tool for summarizing long text documents using LangChain
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain_core.tools import tool
from langchain.docstore.document import Document
from react_agent.config import OPENAI_API_KEY
import os

# Initialize the model
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

@tool
def summarize_text(text: str, style: str = "executive") -> str:
    """
    Summarizes a long text input. Automatically chunks the input.

    Args:
        text (str): The full text to summarize.
        style (str): The tone or format of summary. One of: 'executive', 'detailed', 'bullet_points'

    Returns:
        str: Summarized output.
    """
    if not text or len(text.strip()) < 10:
        return "Input text is too short to summarize."

    # Choose prompt style
    prompt_map = {
        "executive": "Provide an executive summary of the following content:",
        "detailed": "Write a detailed summary with supporting points:",
        "bullet_points": "Summarize the content as a list of concise bullet points:"
    }

    prompt_prefix = prompt_map.get(style.lower(), prompt_map["executive"])

    # Split text into documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.create_documents([text])

    if not docs:
        return "No content was found to summarize."

    # Load chain
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False)
    summary = chain.run(docs)

    return f"{prompt_prefix}\n\n{summary}"
