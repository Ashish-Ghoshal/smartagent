import pytest
from react_agent.new_tools.data_analysis import (
    generate_eda_report,
    auto_llm_insight,
    engineer_features,
    run_pca,
    feature_selection
)

@pytest.mark.asyncio
async def test_eda_report():
    result = generate_eda_report("sales.csv")
    assert "EDA Report" in result

@pytest.mark.asyncio
async def test_llm_insight():
    result = auto_llm_insight("sales.csv")
    assert isinstance(result, str)
    assert "Insight" in result or "❌" not in result

@pytest.mark.asyncio
async def test_feature_engineering():
    result = engineer_features("sales.csv")
    assert "ratio" in result or "❌" not in result

@pytest.mark.asyncio
async def test_pca():
    result = run_pca("sales.csv", n_components=2)
    assert "Explained Variance" in result

@pytest.mark.asyncio
async def test_feature_selection():
    result = feature_selection("sales.csv", threshold=0.01)
    assert "Selected Features" in result
