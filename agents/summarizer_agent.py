"""
summarizer_agent.py – Handles document-based summarization tasks
"""

from agents.base_agent import create_agent
from react_agent.logger import logger
from react_agent.new_tools.document_search import document_search
from react_agent.new_tools.summarize_text import summarize_text
from langchain.agents.tools import Tool

# Define the summarizer tool list
summarizer_tools = [
summarizer_tools = [
    Tool.from_function(
        func=document_search,
        name="DocumentSearch",
        description="Use this to search internal documents and retrieve relevant passages."
    ),
    Tool.from_function(
        func=summarize_text,
        name="SummarizeText",
        description="Use this to summarize long content into executive, detailed, or bullet-point formats."
    )
]
    Tool.from_function(
        func=document_search,
        name="DocumentSearch",
        description="Use this to search internal documents and summarize answers."
    )
]

# Create the summarizer agent using the base factory
summarizer_agent = create_agent(
    tools=summarizer_tools,
    agent_name="SummarizerAgent",
    system_prompt="You are a document summarization expert. Use DocumentSearch to find relevant info and SummarizeText to generate user-friendly summaries. Support styles like executive, detailed, or bullet points. Keep your answers concise unless told otherwise."
    system_prompt="You are a document summarization expert. Use DocumentSearch to find relevant info and SummarizeText to generate user-friendly summaries. Support styles like executive, detailed, or bullet points. Keep your answers concise unless told otherwise."
)
