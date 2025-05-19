"""
logger.py – Central logging utility for SmartAgent
"""

import logging
import os

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure the logger
logger = logging.getLogger("smartagent")
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler("logs/smartagent.log")
file_handler.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach handlers if not already attached
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
