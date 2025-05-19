# SmartAgent Version: v1.0 (Multi-Agent Core)
"""
config.py – Centralized configuration for API keys and settings
"""

import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# API keys and other secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Add additional service keys below as needed
