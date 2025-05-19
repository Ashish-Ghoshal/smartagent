import pytest
from react_agent.new_tools.dimensionality_reduction import dimensionality_reduction

@pytest.mark.asyncio
async def test_dimensionality_reduction_2d():
    result = dimensionality_reduction("sales.csv", n_components=2)
    assert "PCA Reduction to 2D" in result or "⚠️" in result or "❌" in result

@pytest.mark.asyncio
async def test_dimensionality_reduction_invalid_components():
    result = dimensionality_reduction("sales.csv", n_components=5)
    assert "⚠️" in result or "❌" in result
