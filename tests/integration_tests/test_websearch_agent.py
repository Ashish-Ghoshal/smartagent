import pytest
from agents.websearch_agent import websearch_agent
from react_agent.logger import logger

@pytest.mark.asyncio
async def test_websearch_agent_summary():
    query = "What are the recent developments in generative AI as of 2024?"
    logger.info(f"Testing websearch_agent with query: {query}")
    result = websearch_agent.run(query)
    logger.info(f"Result: {result}")
    assert isinstance(result, str)
    assert "web summary" in result.lower() or "⚠️" in result or "🚨" in result
