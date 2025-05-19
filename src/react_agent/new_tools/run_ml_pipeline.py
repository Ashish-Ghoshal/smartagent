"""
run_ml_pipeline.py – Lightweight ML training and prediction tool for SmartAgent
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
from langchain_core.tools import tool

DATA_DIR = "data/tables"

@tool
def run_ml_pipeline(filename: str, target: str, task: str = "classification") -> str:
    """
    Trains a lightweight ML model on a CSV file.

    Args:
        filename (str): CSV filename inside /data/tables/
        target (str): Target column name
        task (str): 'classification', 'regression', or 'clustering'

    Returns:
        str: Summary of model performance
    """
    try:
        path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(path)

        if task == "clustering":
            numeric = df.select_dtypes(include='number').dropna()
            if numeric.shape[1] < 2:
                return "⚠️ Not enough numeric features for clustering."
            kmeans = KMeans(n_clusters=3, n_init='auto')
            labels = kmeans.fit_predict(numeric)
            score = silhouette_score(numeric, labels)
            return f"✅ KMeans Silhouette Score: {score:.3f}"

        if target not in df.columns:
            return f"⚠️ Target column '{target}' not found."

        X = df.drop(columns=[target])
        y = df[target]

        # Auto-select numeric or one-hot encode for simplicity
        X = pd.get_dummies(X, drop_first=True)
        y = y if task == "regression" else pd.Series(y).astype('category').cat.codes

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if task == "classification":
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            return f"✅ RandomForest Classification Accuracy: {acc:.3f}"

        elif task == "regression":
            model = LinearRegression()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            return f"✅ Linear Regression MSE: {mse:.3f}"

        else:
            return f"❌ Unknown task: '{task}'"

    except Exception as e:
        return f"❌ Error in ML pipeline: {e}"
