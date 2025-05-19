"""
explain_clusters.py – Use LLM to describe each cluster based on group statistics
"""

import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.tools import tool
from react_agent.config import OPENAI_API_KEY

DATA_DIR = "data/tables"

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

explanation_prompt = PromptTemplate(
    input_variables=["cluster_summary"],
    template="""You are a data analyst. Given the cluster summaries below, explain what each group represents in plain English.
Avoid technical jargon and keep it concise and insightful.

Cluster Summary:
{cluster_summary}

Respond with labeled bullet points, one per cluster.
"""
)
llm_chain = LLMChain(llm=llm, prompt=explanation_prompt)

@tool
def explain_clusters(filename: str, n_clusters: int = 3) -> str:
    """
    Uses KMeans to cluster the dataset, summarizes each group, and lets the LLM explain what each cluster represents.

    Args:
        filename (str): CSV filename
        n_clusters (int): Number of clusters to use

    Returns:
        str: LLM-generated explanation of each cluster
    """
    try:
        path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(path)
        numeric_df = df.select_dtypes(include='number').dropna()

        if numeric_df.shape[1] < 2:
            return "⚠️ Not enough numeric features to cluster."

        X_scaled = StandardScaler().fit_transform(numeric_df)
        model = KMeans(n_clusters=n_clusters, n_init='auto')
        labels = model.fit_predict(X_scaled)

        df["cluster"] = labels
        summary = df.groupby("cluster").agg(["mean", "median", "count"]).round(2)
        summary_text = summary.to_string()

        explanation = llm_chain.run(cluster_summary=summary_text).strip()
        return f"🧠 Cluster Explanation (based on {n_clusters} groups):\n\n{explanation}"

    except Exception as e:
        return f"❌ explain_clusters failed: {e}"
