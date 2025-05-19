"""
smart_clustering_selector.py – LLM-guided clustering tool with Silhouette evaluation
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.tools import tool
from react_agent.config import OPENAI_API_KEY

DATA_DIR = "data/tables"

# LLM to recommend clustering method
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
cluster_prompt = PromptTemplate(
    input_variables=["n_samples", "n_features"],
    template="""You are a clustering expert. Recommend the most suitable clustering algorithms for a dataset with:

Samples: {n_samples}
Features: {n_features}

Return 2-3 algorithms from this list:
- KMeans
- DBSCAN
- AgglomerativeClustering

Respond with a comma-separated list of model names only.
"""
)
cluster_chain = LLMChain(llm=llm, prompt=cluster_prompt)

@tool
def smart_clustering_selector(filename: str) -> str:
    """
    Uses LLM to suggest clustering algorithms and evaluates them using silhouette score.

    Args:
        filename (str): CSV filename

    Returns:
        str: Best clustering result with score
    """
    try:
        path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(path)
        df_numeric = df.select_dtypes(include='number').dropna()
        if df_numeric.shape[0] < 10 or df_numeric.shape[1] < 2:
            return "⚠️ Not enough numeric data for clustering."

        # Standardize
        X = StandardScaler().fit_transform(df_numeric)
        n_samples, n_features = X.shape

        suggestion = cluster_chain.run(n_samples=n_samples, n_features=n_features).strip()
        models = [m.strip() for m in suggestion.split(",") if m.strip()]

        results = []
        for name in models:
            try:
                if name == "KMeans":
                    for k in [2, 3, 4]:
                        km = KMeans(n_clusters=k, n_init='auto')
                        labels = km.fit_predict(X)
                        score = silhouette_score(X, labels)
                        results.append((f"KMeans (k={k})", score))
                elif name == "DBSCAN":
                    db = DBSCAN()
                    labels = db.fit_predict(X)
                    if len(set(labels)) > 1:
                        score = silhouette_score(X, labels)
                        results.append(("DBSCAN", score))
                elif name == "AgglomerativeClustering":
                    ac = AgglomerativeClustering(n_clusters=3)
                    labels = ac.fit_predict(X)
                    score = silhouette_score(X, labels)
                    results.append(("AgglomerativeClustering", score))
            except Exception as e:
                results.append((name, f"error: {e}"))

        sorted_results = sorted(results, key=lambda x: x[1] if isinstance(x[1], (int, float)) else -999, reverse=True)
        report = ["✅ Clustering Results (Silhouette Scores):"]
        for name, score in sorted_results:
            report.append(f"- {name}: {score}")
        return "\n".join(report)

    except Exception as e:
        return f"❌ Clustering failed: {e}"
