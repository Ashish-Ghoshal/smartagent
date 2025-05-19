"""
websearch_agent.py – Enhanced agent for real-time search with summarization, style control, and source listing
"""

from agents.base_agent import create_agent
from react_agent.new_tools.web_search import web_search
from react_agent.new_tools.summarize_text import summarize_text
from langchain.agents.tools import Tool
from react_agent.logger import logger

def smart_web_summary(query: str, style: str = "executive") -> str:
    """
    Combines real-time web search with summarization for clean output.

    Args:
        query (str): User's query to search and summarize
        style (str): Summary format style: executive, detailed, bullet_points

    Returns:
        str: Clean answer with sources or a fallback explanation
    """
    try:
        search_results = web_search.invoke(query)
        if not search_results or "not configured" in search_results.lower():
            logger.warning("Web search tool returned empty or error.")
            return "⚠️ Web search could not be performed. Please check your API key or try later."

        source_count = search_results.count("http")  # crude but effective
        logger.info(f"Retrieved {source_count} sources from Tavily.")

        summary = summarize_text.invoke({
            "text": search_results,
            "style": style
        })

        logger.info("Web search and summarization completed.")
        return f"📡 Web Summary ({style.title()}):

{summary}

🔗 Based on {source_count} sources."
    
    except Exception as e:
        logger.error(f"WebSearch agent failed: {e}")
        return f"🚨 Error processing web search: {e}"

# Register updated tool
websearch_tools = [
    Tool.from_function(
        func=smart_web_summary,
        name="SmartWebSummary",
        description="Search and summarize with sources. Supports style: executive, detailed, bullet_points."
    )
]

websearch_agent = create_agent(
    tools=websearch_tools,
    agent_name="WebSearchAgent",
    system_prompt="You are a real-time web search assistant. Use SmartWebSummary to find the latest information, summarize it concisely, and mention how many sources were used. Be clear and direct. Default to executive summaries unless specified."
)
