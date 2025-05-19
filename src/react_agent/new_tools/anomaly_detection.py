"""
anomaly_detection.py – Detect outliers using IsolationForest and LocalOutlierFactor
"""

import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from langchain_core.tools import tool

DATA_DIR = "data/tables"

@tool
def detect_anomalies(filename: str, method: str = "isolation", n_top: int = 5) -> str:
    """
    Detects outliers using Isolation Forest or Local Outlier Factor.

    Args:
        filename (str): CSV file in /data/tables/
        method (str): 'isolation' or 'lof'
        n_top (int): Number of top anomalies to display

    Returns:
        str: Summary of top anomalies with scores
    """
    try:
        if method not in ["isolation", "lof"]:
            return "⚠️ Method must be 'isolation' or 'lof'."

        path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(path)
        numeric = df.select_dtypes(include='number').dropna()

        if numeric.shape[0] < 10 or numeric.shape[1] < 2:
            return "⚠️ Not enough numeric data for anomaly detection."

        X = StandardScaler().fit_transform(numeric)

        if method == "isolation":
            model = IsolationForest(contamination=0.1, random_state=42)
            scores = model.fit_predict(X)
            anomaly_scores = model.decision_function(X) * -1  # higher = more anomalous
        else:  # LOF
            model = LocalOutlierFactor(n_neighbors=20)
            scores = model.fit_predict(X)
            anomaly_scores = model.negative_outlier_factor_ * -1

        df_anomalies = df.copy()
        df_anomalies["anomaly_score"] = anomaly_scores
        df_top = df_anomalies.sort_values("anomaly_score", ascending=False).head(n_top)
        preview = df_top[["anomaly_score"] + list(numeric.columns)].to_markdown(index=False)

        return f"🚨 Top {n_top} Anomalies Detected ({method}):\n\n{preview}"

    except Exception as e:
        return f"❌ Anomaly detection failed: {e}"
