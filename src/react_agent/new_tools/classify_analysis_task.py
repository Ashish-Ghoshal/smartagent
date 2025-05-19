"""
classify_analysis_task.py – Use LLM to route analysis queries to the correct tool based on CSV metadata and user query
"""

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from react_agent.config import OPENAI_API_KEY
import os
import pandas as pd
from langchain_core.tools import tool

DATA_DIR = "data/tables"

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

prompt = PromptTemplate(
    input_variables=["query", "columns", "dtypes"],
    template="""You are a tool classifier for a data analyst agent. Based on the user's question and the CSV structure, 
choose the best tool to use. Only return one tool name from this list:

- GenerateEDAReport
- LLMInsight
- RunPCA
- FeatureSelection
- EngineerFeatures
- JoinTables
- Unsupported

User Query: {query}
CSV Columns: {columns}
Column Types: {dtypes}

Respond only with the tool name above.
"""
)

classifier_chain = LLMChain(llm=llm, prompt=prompt)

@tool
def classify_analysis_task(query: str, filename: str) -> str:
    """
    Determines the best analysis tool for a given CSV and query.

    Args:
        query (str): User question or task
        filename (str): CSV filename inside /data/tables/

    Returns:
        str: Tool name or 'Unsupported'
    """
    try:
        filepath = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(filepath)
        cols = list(df.columns)
        types = df.dtypes.astype(str).to_dict()
        response = classifier_chain.run(query=query, columns=cols, dtypes=str(types)).strip()
        return f"✅ Recommended Tool: {response}"
    except Exception as e:
        return f"❌ Could not classify task: {e}"
