"""
smart_controller_memory.py – SmartAgent controller with session memory support
"""

from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from react_agent.agents.analyst_agent import analyst_tools
from react_agent.config import OPENAI_API_KEY

# Enable session memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

smart_agent_with_memory = initialize_agent(
    tools=analyst_tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)
