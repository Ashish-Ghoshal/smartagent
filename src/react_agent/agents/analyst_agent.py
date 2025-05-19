"""
analyst_agent.py – Handles tasks involving numerical or tabular data analysis
"""

from agents.base_agent import create_agent
from react_agent.new_tools.data_analysis import (
    generate_eda_report,
    auto_llm_insight,
    join_tables,
    engineer_features,
    run_pca,
    feature_selection
)
from react_agent.new_tools.run_ml_pipeline import run_ml_pipeline
from react_agent.new_tools.smart_model_selector import smart_model_selector
from react_agent.new_tools.smart_clustering_selector import smart_clustering_selector
from react_agent.new_tools.dimensionality_reduction import dimensionality_reduction
from react_agent.new_tools.anomaly_detection import detect_anomalies
from react_agent.new_tools.explain_clusters import explain_clusters
from langchain.agents.tools import Tool
from react_agent.logger import logger

# Define the tools available to the analyst agent
analyst_tools = [
    Tool.from_function(
        func=smart_clustering_selector,
        name="SmartClusteringSelector",
        description="Uses LLM to suggest clustering algorithms and evaluate them using silhouette scores."
    ),
    Tool.from_function(
        func=smart_model_selector,
        name="SmartModelSelector",
        description="Uses LLM to suggest and evaluate ML models for classification or regression."
    ),
    Tool.from_function(
        func=run_ml_pipeline,
        name="RunMLPipeline",
        description="Trains a simple ML model on classification, regression, or clustering tasks."
    ),
    Tool.from_function(
        func=generate_eda_report,
        name="GenerateEDAReport",
        description="Performs detailed exploratory data analysis on a single CSV file."
    ),
    Tool.from_function(
        func=auto_llm_insight,
        name="LLMInsight",
        description="Explains trends and anomalies in a dataset using AI-based analysis."
    ),
    Tool.from_function(
        func=join_tables,
        name="JoinTables",
        description="Merges two CSV files on a specified column."
    ),
    Tool.from_function(
        func=engineer_features,
        name="EngineerFeatures",
        description="Creates engineered features like ratios and totals from a CSV."
    ),
    Tool.from_function(
        func=run_pca,
        name="RunPCA",
        description="Performs PCA on numeric columns to reduce dimensionality."
    ),
    Tool.from_function(
        func=feature_selection,
        name="FeatureSelection",
        description="Selects high-variance numeric columns for further modeling."
    ),
    Tool.from_function(
        func=dimensionality_reduction,
        name="DimensionalityReduction",
        description="Reduces dataset to 2D or 3D using PCA and returns a preview and variance explained."
    ),
    Tool.from_function(
        func=detect_anomalies,
        name="DetectAnomalies",
        description="Detects outliers using Isolation Forest or Local Outlier Factor (LOF)."
    ),
    Tool.from_function(
        func=explain_clusters,
        name="ExplainClusters",
        description="Uses KMeans and LLM to describe the meaning of each cluster in plain English."
    )
]

analyst_agent = create_agent(
    tools=analyst_tools,
    agent_name="AnalystAgent",
    system_prompt="You are a data analyst. Use the tools provided to perform EDA, clustering, feature engineering, model training, PCA, anomaly detection, dimensionality reduction, or table operations. Be precise and answer in markdown when appropriate."
)
