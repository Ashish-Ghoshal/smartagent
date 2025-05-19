import pytest
from react_agent.new_tools.table_analysis import table_analysis

@pytest.mark.asyncio
async def test_table_analysis_mean():
    result = table_analysis("test_data.csv", "price", "mean")
    assert "Mean of 'price'" in result

@pytest.mark.asyncio
async def test_table_analysis_sum():
    result = table_analysis("test_data.csv", "quantity", "sum")
    assert "Sum of 'quantity'" in result

@pytest.mark.asyncio
async def test_table_analysis_missing_column():
    result = table_analysis("test_data.csv", "nonexistent", "mean")
    assert "Column 'nonexistent' not found" in result

@pytest.mark.asyncio
async def test_table_analysis_bad_file():
    result = table_analysis("missing.csv", "price", "mean")
    assert "not found" in result
