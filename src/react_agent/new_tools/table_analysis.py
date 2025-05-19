"""
table_analysis.py – Tool to analyze tabular business data (CSV)
"""

import os
import pandas as pd
from langchain_core.tools import tool

DATA_DIR = "data/tables"

@tool
def table_analysis(filename: str, column: str, operation: str) -> str:
    """
    Analyze a column in a CSV file and return a summary statistic.

    Args:
        filename (str): Name of the CSV file (must be in data/tables/)
        column (str): Name of the column to analyze
        operation (str): One of: mean, sum, max, min, count

    Returns:
        str: The result of the analysis or an error message
    """
    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        return f"File '{filename}' not found in {DATA_DIR}."

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return f"Failed to read file: {e}"

    if column not in df.columns:
        return f"Column '{column}' not found in {filename}."

    try:
        result = {
            "mean": df[column].mean(),
            "sum": df[column].sum(),
            "max": df[column].max(),
            "min": df[column].min(),
            "count": df[column].count()
        }.get(operation.lower(), "Unsupported operation.")
    except Exception as e:
        return f"Error processing column: {e}"

    return f"{operation.capitalize()} of '{column}' in '{filename}': {result}"
