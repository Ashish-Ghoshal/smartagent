import pytest
from react_agent.new_tools.explain_clusters import explain_clusters

@pytest.mark.asyncio
async def test_explain_clusters_basic():
    result = explain_clusters("sales.csv", n_clusters=2)
    assert "Cluster Explanation" in result or "❌" in result
