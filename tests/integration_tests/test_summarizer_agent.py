import pytest
from agents.summarizer_agent import summarizer_agent

@pytest.mark.asyncio
async def test_document_search_summary():
    result = summarizer_agent.run("Search and summarize any documents related to Stripe pricing.")
    assert isinstance(result, str)
    assert len(result) > 0

@pytest.mark.asyncio
async def test_summarize_text_executive():
    long_text = "SmartAgent is an AI assistant framework designed to handle various tasks via sub-agents. Each sub-agent is responsible for a specific domain, such as summarization, web search, or data analysis. The summarizer agent can use tools to condense long documents, while the analyst agent is used for interpreting CSV data."
    result = summarizer_agent.run(f"Summarize the following in executive style: {long_text}")
    assert isinstance(result, str)
    assert "summary" in result.lower() or len(result) > 0

@pytest.mark.asyncio
async def test_summarize_text_bullet_points():
    text = "LangChain is a powerful library that enables LLM orchestration. It helps in building autonomous agents that can plan and act. It supports tool use, memory, and even multi-agent collaboration."
    result = summarizer_agent.run(f"Summarize this as bullet points: {text}")
    assert isinstance(result, str)
    assert "-" in result or "*" in result  # bullet points format
