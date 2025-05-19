import pytest
from agents.controller_agent import route_task

@pytest.mark.asyncio
async def test_route_summarization():
    result = route_task.invoke("Summarize the quarterly performance report for Stripe.")
    assert isinstance(result, str)
    assert len(result) > 0

@pytest.mark.asyncio
async def test_route_analysis():
    result = route_task.invoke("Compare revenue across months in the uploaded CSV.")
    assert isinstance(result, str)
    assert len(result) > 0

@pytest.mark.asyncio
async def test_route_web_search():
    result = route_task.invoke("What is the latest update on OpenAI's funding?")
    assert isinstance(result, str)
    assert len(result) > 0

@pytest.mark.asyncio
async def test_route_unknown():
    result = route_task.invoke("Sing me a song.")
    assert isinstance(result, str)
    assert "could not classify" in result.lower() or "⚠️" in result or "sorry" in result.lower()
