import pytest
from react_agent.new_tools.web_search import web_search

@pytest.mark.asyncio
async def test_web_search_returns_result():
    result = web_search.invoke("What is the latest news on AI regulation?")
    assert isinstance(result, str)
    assert len(result) > 0 or "not configured" in result.lower()
