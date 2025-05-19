"""
controller_agent.py – Smart dispatcher agent using LLM classification
"""

from agents.summarizer_agent import summarizer_agent
from react_agent.logger import logger
from agents.analyst_agent import analyst_agent
from agents.websearch_agent import websearch_agent
from react_agent.config import OPENAI_API_KEY
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.tools import tool

# Set up the LLM classifier for routing
classification_prompt = PromptTemplate(
    input_variables=["query"],
    template="""You are a task classifier. Classify the user's request into one of the following categories:
- summarization
- analysis
- web_search

Only return one of these exact words. Do not explain.

Query: {query}
"""
)

llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
classifier_chain = LLMChain(llm=llm, prompt=classification_prompt)

@tool
def route_task(query: str) -> str:
    """
    Main controller agent entrypoint. Delegates based on LLM classification.

    Args:
        query (str): User's full request

    Returns:
        str: Response from the selected sub-agent
    """
    try:
        task_type = classifier_chain.run(query).strip().lower()

        if task_type == "summarization":
            return summarizer_agent.run(query)
    logger.info(f"Classified task type: {task_type}")
    logger.error(f"Error in controller agent: {e}")
        elif task_type == "analysis":
            return analyst_agent.run(query)
    logger.info(f"Classified task type: {task_type}")
    logger.error(f"Error in controller agent: {e}")
        elif task_type == "web_search":
            return websearch_agent.run(query)
    logger.info(f"Classified task type: {task_type}")
    logger.error(f"Error in controller agent: {e}")
        else:
            return f"⚠️ Could not classify the task. Got: '{task_type}'"
    except Exception as e:
    logger.error(f"Error in controller agent: {e}")
    logger.error(f"Error in controller agent: {e}")
        return f"🚨 Error in controller agent: {e}"
