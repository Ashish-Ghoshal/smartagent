# SmartAgent Version: v1.3 (WebSearch + Smart Summary)
"""
graph.py – Entrypoint for SmartAgent using controller agent
"""

from agents.controller_agent import route_task
from react_agent.logger import logger

def run_smartagent():
    logger.info("SmartAgent started. Waiting for user input...")
    print("Welcome to SmartAgent! Type 'exit' to quit.")
    print("You can ask questions about documents, data analysis, or current events.")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        logger.info(f"User query: {user_input}")
        result = route_task.invoke(user_input)
        logger.info(f"Agent response: {result}")
        print(f"\nAgent: {result}")

if __name__ == "__main__":
    run_smartagent()
