"""Shared API settings for the tutorial notebook and Python script.

Replace the placeholders with the values supplied by your API provider.
Restore the API key placeholder before sharing or pushing repository changes.
"""

# Choose one: "openai", "anthropic", or "openai_compatible".
PROVIDER = "openai"

# Replace with the exact model ID that your API key can access.
MODEL = "gpt5.6-luna light"

# Paste your API key here, between the quotes.
API_KEY = "PASTE_API_KEY_HERE"

# Leave blank for standard OpenAI or Anthropic APIs. For a school endpoint,
# use PROVIDER = "openai_compatible" and paste its URL here.
BASE_URL = ""

MAX_MODEL_CALLS = 35
