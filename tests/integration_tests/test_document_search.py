import pytest
from react_agent.new_tools.document_search import document_search

@pytest.mark.asyncio
async def test_document_search_returns_result():
    result = document_search.invoke("sample query about pricing")
    assert isinstance(result, str)
    assert len(result) > 0 or "no relevant documents" in result.lower()
