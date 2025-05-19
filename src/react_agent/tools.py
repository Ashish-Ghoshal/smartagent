"""
tools.py – Register tools available to the agent
"""

from langchain_core.tools import Tool
from react_agent.new_tools.document_search import document_search
from react_agent.new_tools.web_search import web_search
from react_agent.new_tools.table_analysis import table_analysis

TOOLS = [
    document_search,
    web_search,
    table_analysis,
]
