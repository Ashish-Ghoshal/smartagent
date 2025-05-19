"""
dimensionality_reduction.py – Reduce high-dimensional datasets using PCA
"""

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from langchain_core.tools import tool

DATA_DIR = "data/tables"

@tool
def dimensionality_reduction(filename: str, n_components: int = 2) -> str:
    """
    Applies PCA to reduce dataset to 2 or 3 dimensions for visualization.

    Args:
        filename (str): CSV filename inside /data/tables/
        n_components (int): Number of dimensions (2 or 3)

    Returns:
        str: Markdown-style summary of reduced dimensions
    """
    try:
        if n_components not in [2, 3]:
            return "⚠️ Please choose 2 or 3 components only."

        path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(path)
        numeric_df = df.select_dtypes(include='number').dropna()

        if numeric_df.shape[1] < n_components:
            return "⚠️ Not enough numeric columns for PCA."

        X_scaled = StandardScaler().fit_transform(numeric_df)
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(X_scaled)

        result_df = pd.DataFrame(reduced, columns=[f"PC{i+1}" for i in range(n_components)])
        result_df["index"] = df.index
        preview = result_df.head().to_markdown(index=False)

        explained = pca.explained_variance_ratio_
        explained_str = ", ".join(f"{v:.2%}" for v in explained)

        return f"✅ PCA Reduction to {n_components}D:\nExplained Variance: {explained_str}\n\nPreview:\n{preview}"

    except Exception as e:
        return f"❌ PCA failed: {e}"
