"""
web_search.py – Tool for web search using Tavily API
"""

import os
import requests
from langchain_core.tools import tool
from react_agent.config import TAVILY_API_KEY

TAVILY_ENDPOINT = "https://api.tavily.com/search"

@tool
def web_search(query: str) -> str:
    """
    Perform a real-time web search and return a formatted summary of results.

    Args:
        query (str): Search query

    Returns:
        str: Combined answer with key links and snippets
    """
    if not TAVILY_API_KEY:
        return "Tavily API key not configured. Please set TAVILY_API_KEY in your .env file."

    response = requests.post(
        TAVILY_ENDPOINT,
        headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
        json={"query": query, "search_depth": "basic", "include_answer": True}
    )

    if response.status_code != 200:
        return f"Web search failed with status code {response.status_code}"

    data = response.json()

    answer = data.get("answer", "").strip()
    results = data.get("results", [])

    if not results:
        return answer if answer else "No search results found."

    # Format sources
    formatted_sources = []
    for r in results[:5]:  # Limit to top 5
        snippet = r.get("content", "").strip()
        url = r.get("url", "").strip()
        title = r.get("title", "Source").strip()
        if snippet and url:
            formatted_sources.append(f"🔗 {title}
{snippet}
{url}")

    sources_text = "\n\n".join(formatted_sources)

    return f"{answer}\n\nSources:\n{sources_text}"
