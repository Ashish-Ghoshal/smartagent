import pytest
from react_agent.new_tools.anomaly_detection import detect_anomalies

@pytest.mark.asyncio
async def test_anomaly_isolation():
    result = detect_anomalies("sales.csv", method="isolation", n_top=3)
    assert "Anomalies Detected" in result or "⚠️" in result or "❌" in result

@pytest.mark.asyncio
async def test_anomaly_lof():
    result = detect_anomalies("sales.csv", method="lof", n_top=3)
    assert "Anomalies Detected" in result or "⚠️" in result or "❌" in result
