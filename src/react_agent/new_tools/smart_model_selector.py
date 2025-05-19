"""
smart_model_selector.py – LLM-guided model selection and optional tuning for classification and regression
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from langchain.chat_models import ChatOpenAI
from langchain_core.tools import tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from react_agent.config import OPENAI_API_KEY

DATA_DIR = "data/tables"

# LLM model selector
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
selector_prompt = PromptTemplate(
    input_variables=["target_type", "n_samples", "n_features"],
    template="""You are a machine learning model selector.
Given the data shape and target type, suggest 2-4 models for this task.

Target type: {target_type}
Samples: {n_samples}
Features: {n_features}

Return a comma-separated list of model names from this set:
- LogisticRegression
- RandomForestClassifier
- GradientBoostingClassifier
- DecisionTreeClassifier
- LinearRegression
- Ridge
- RandomForestRegressor
"""
)
selector_chain = LLMChain(llm=llm, prompt=selector_prompt)

def evaluate_model(model, task, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    if task == "classification":
        return accuracy_score(y_test, preds)
    else:
        return mean_squared_error(y_test, preds)

@tool
def smart_model_selector(filename: str, target: str, task: str = "classification", tune: bool = None) -> str:
    """
    Uses LLM to suggest and evaluate models, with optional tuning based on data size.

    Args:
        filename (str): CSV filename
        target (str): Column to predict
        task (str): 'classification' or 'regression'
        tune (bool): Whether to perform light tuning (auto if None)

    Returns:
        str: Sorted list of models and their scores
    """
    try:
        path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(path)

        if target not in df.columns:
            return f"⚠️ Target '{target}' not found."

        y = df[target]
        X = df.drop(columns=[target])
        X = pd.get_dummies(X, drop_first=True)
        if task == "classification":
            y = y.astype("category").cat.codes

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        n_samples, n_features = X_train.shape

        # Auto-set tuning if not specified
        if tune is None:
            tune = n_samples < 2000 and n_features < 20

        suggestion = selector_chain.run(
            target_type=task,
            n_samples=n_samples,
            n_features=n_features
        ).strip()
        model_names = [m.strip() for m in suggestion.split(",") if m.strip()]
        if not model_names:
            return "❌ LLM failed to suggest models."

        results = []
        for name in model_names:
            try:
                model = eval(f"{name}()")
                if tune:
                    param_grid = {
                        "LogisticRegression": {"C": [0.1, 1.0, 10], "penalty": ["l2"]},
                        "RandomForestClassifier": {"n_estimators": [50, 100], "max_depth": [None, 10]},
                        "GradientBoostingClassifier": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
                        "DecisionTreeClassifier": {"max_depth": [None, 5, 10]},
                        "Ridge": {"alpha": [0.5, 1.0, 2.0]},
                        "RandomForestRegressor": {"n_estimators": [50, 100], "max_depth": [None, 10]},
                        "LinearRegression": {}
                    }
                    grid = param_grid.get(name, {})
                    if grid:
                        model = RandomizedSearchCV(model, param_distributions=grid, n_iter=3, cv=3, random_state=42)
                score = evaluate_model(model, task, X_train, X_test, y_train, y_test)
                results.append((name, score))
            except Exception as e:
                results.append((name, f"error: {e}"))

        sorted_results = sorted(results, key=lambda x: x[1] if isinstance(x[1], (int, float)) else float("inf"))
        report = [f"✅ Model Test Results (Tuning={'enabled' if tune else 'disabled'}):"]
        for name, score in sorted_results:
            report.append(f"- {name}: {score}")

        return "\n".join(report)

    except Exception as e:
        return f"❌ smart_model_selector failed: {e}"
