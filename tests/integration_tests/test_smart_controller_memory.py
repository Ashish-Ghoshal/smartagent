import pytest
from react_agent.agents.smart_controller_memory import smart_agent_with_memory

@pytest.mark.asyncio
async def test_memory_agent_basic_query():
    try:
        response = smart_agent_with_memory.run("Please generate an EDA report for sales.csv")
        assert "EDA" in response or "❌" not in response
    except Exception as e:
        assert False, f"Agent memory test failed: {e}"
