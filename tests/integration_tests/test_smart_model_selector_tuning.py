import pytest
from react_agent.new_tools.smart_model_selector import smart_model_selector

@pytest.mark.asyncio
async def test_selector_with_tuning():
    result = smart_model_selector("sales.csv", target="Region", task="classification", tune=True)
    assert "Model Test Results" in result
    assert "Tuning=enabled" in result

@pytest.mark.asyncio
async def test_selector_without_tuning():
    result = smart_model_selector("sales.csv", target="Sales", task="regression", tune=False)
    assert "Model Test Results" in result
    assert "Tuning=disabled" in result
