"""
data_analysis.py – Exploratory data analysis with insights using Pandas and optional LLM
"""

import os
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain_core.tools import tool
from react_agent.config import OPENAI_API_KEY

DATA_DIR = "data/tables"

# Initialize OpenAI LLM for insights (optional)
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

def load_csv(filename):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filename}' not found in {DATA_DIR}.")
    return pd.read_csv(filepath)

@tool
def generate_eda_report(filename: str) -> str:
    """
    Generates a detailed EDA report including types, nulls, basic stats, and unique values.

    Args:
        filename (str): CSV filename inside /data/tables/

    Returns:
        str: Markdown-formatted EDA summary
    """
    try:
        df = load_csv(filename)
        report = []

        report.append(f"# 📊 EDA Report for `{filename}`")
        report.append(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

        report.append("\n## 🧱 Column Types")
        report.append(df.dtypes.to_markdown())

        report.append("\n## ❓ Missing Values")
        report.append(df.isnull().sum().to_markdown())

        report.append("\n## 🔢 Summary Statistics")
        report.append(df.describe(include='all').fillna("").to_markdown())

        report.append("\n## 🔁 Unique Values Per Column")
        report.append(df.nunique().to_markdown())

        return "\n".join(report)

    except Exception as e:
        return f"❌ Failed to generate EDA: {e}"

@tool
def auto_llm_insight(filename: str) -> str:
    """
    Runs LLM on top of EDA stats to provide a natural language interpretation.

    Args:
        filename (str): CSV filename inside /data/tables/

    Returns:
        str: LLM-generated insight
    """
    try:
        df = load_csv(filename)
        stats_md = df.describe(include='all').fillna("").to_markdown()
        query = f"Here is a summary of a dataset:\n{stats_md}\n\nPlease explain key trends and any anomalies in plain English."

        result = llm.predict(query)
        return f"🧠 Insight Summary:\n\n{result.strip()}"

    except Exception as e:
        return f"❌ LLM insight failed: {e}"

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

@tool
def join_tables(file1: str, file2: str, on: str, how: str = "inner") -> str:
    """
    Merge two CSV files on a shared column.

    Args:
        file1 (str): First CSV filename
        file2 (str): Second CSV filename
        on (str): Column to join on
        how (str): Join type (inner, left, right, outer)

    Returns:
        str: Summary of merged DataFrame
    """
    try:
        df1 = load_csv(file1)
        df2 = load_csv(file2)
        merged = pd.merge(df1, df2, on=on, how=how)
        return f"✅ Merged shape: {merged.shape}\nColumns: {list(merged.columns)}"
    except Exception as e:
        return f"❌ Merge failed: {e}"

@tool
def engineer_features(filename: str) -> str:
    """
    Adds common engineered features to a CSV, such as row sums, ratios, and categoricals.

    Args:
        filename (str): CSV filename

    Returns:
        str: Modified columns or results
    """
    try:
        df = load_csv(filename)
        numeric = df.select_dtypes(include='number')
        if numeric.shape[1] < 2:
            return "⚠️ Not enough numeric columns to create ratios."

        col1, col2 = numeric.columns[:2]
        df["ratio_" + col1 + "_to_" + col2] = df[col1] / (df[col2] + 1e-9)
        df["total_sum"] = numeric.sum(axis=1)

        return df[[col1, col2, "ratio_" + col1 + "_to_" + col2, "total_sum"]].head().to_markdown()
    except Exception as e:
        return f"❌ Feature engineering failed: {e}"

@tool
def run_pca(filename: str, n_components: int = 2) -> str:
    """
    Perform PCA on numeric columns.

    Args:
        filename (str): CSV filename
        n_components (int): Number of principal components to return

    Returns:
        str: PCA explained variance summary
    """
    try:
        df = load_csv(filename)
        numeric = df.select_dtypes(include='number').dropna()
        pca = PCA(n_components=n_components)
        pca.fit(numeric)

        result = ["# 📉 PCA Result"]
        result.append(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
        return "\n".join(result)
    except Exception as e:
        return f"❌ PCA failed: {e}"

@tool
def feature_selection(filename: str, threshold: float = 0.01) -> str:
    """
    Select features with variance above threshold.

    Args:
        filename (str): CSV filename
        threshold (float): Minimum variance to keep

    Returns:
        str: Selected feature names
    """
    try:
        df = load_csv(filename)
        numeric = df.select_dtypes(include='number').dropna()
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(numeric)

        kept = numeric.columns[selector.get_support()].tolist()
        return f"✅ Selected Features: {kept}"
    except Exception as e:
        return f"❌ Feature selection failed: {e}"
