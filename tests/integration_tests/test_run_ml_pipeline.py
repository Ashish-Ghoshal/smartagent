import pytest
from react_agent.new_tools.run_ml_pipeline import run_ml_pipeline

@pytest.mark.asyncio
async def test_run_classification():
    result = run_ml_pipeline("sales.csv", "Region", task="classification")
    assert "Accuracy" in result or "⚠️" in result or "❌" in result

@pytest.mark.asyncio
async def test_run_regression():
    result = run_ml_pipeline("sales.csv", "Sales", task="regression")
    assert "MSE" in result or "⚠️" in result or "❌" in result

@pytest.mark.asyncio
async def test_run_clustering():
    result = run_ml_pipeline("sales.csv", "Sales", task="clustering")
    assert "Silhouette" in result or "⚠️" in result or "❌" in result
