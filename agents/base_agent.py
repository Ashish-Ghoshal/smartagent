"""
base_agent.py – Shared logic for creating and configuring LLM agents
"""

from react_agent.config import OPENAI_API_KEY
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.agents.tools import Tool
from langchain.memory import ConversationBufferMemory

def create_agent(tools: list[Tool], agent_name: str, system_prompt: str = ""):
    """
    Initializes a basic agent with memory, system prompt, and tool binding.

    Args:
        tools (list): List of LangChain tools to bind
        agent_name (str): Identifier for the agent
        system_prompt (str): Optional custom system prompt

    Returns:
        AgentExecutor: A LangChain agent executor
    """
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent
