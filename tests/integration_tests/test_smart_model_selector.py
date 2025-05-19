import pytest
from react_agent.new_tools.smart_model_selector import smart_model_selector

@pytest.mark.asyncio
async def test_smart_model_selector_classification():
    result = smart_model_selector("sales.csv", target="Region", task="classification")
    assert "Model Test Results" in result or "❌" not in result

@pytest.mark.asyncio
async def test_smart_model_selector_regression():
    result = smart_model_selector("sales.csv", target="Sales", task="regression")
    assert "Model Test Results" in result or "❌" not in result
