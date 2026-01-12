"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# Poe API key - obtain from poe.com/api_key
POE_API_KEY = os.getenv("POE_API_KEY")

# Council members - list of Poe.com bot display names
# Available bots include: GPT-5, GPT-4o, GPT-4o-Mini, Claude-Sonnet-4.5, 
# Claude-3-Haiku, Gemini-2.5-Pro, Gemini-2.0-Flash, Grok-4, etc.
# See poe.com for the full list of available bots
COUNCIL_MODELS = [
    "GPT-5",
    "Gemini-2.5-Pro",
    "Claude-Sonnet-4.5",
    "Grok-4",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "Gemini-2.5-Pro"

# Title generation model - fast/cheap model for generating conversation titles
TITLE_MODEL = "GPT-4o-Mini"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
