"""Local API settings for the tutorial.

Copy this file to api_config.py, then replace the values below with the values
provided by the school or your API provider. api_config.py is ignored by Git.
"""

# Choose one: "openai", "anthropic", or "openai_compatible".
PROVIDER = "openai"

# Example only. Replace this with the exact model ID you are authorized to use.
MODEL = "gpt-4.1-mini"

# Paste the API key between the quotes. Do not commit this value.
API_KEY = "PASTE_API_KEY_HERE"

# Leave blank for the standard OpenAI or Anthropic API.
# For a school OpenAI-compatible endpoint, set PROVIDER to "openai_compatible"
# and paste the endpoint URL here, for example: "https://example.edu/v1".
BASE_URL = ""

# Maximum number of model calls allowed during one notebook session.
MAX_MODEL_CALLS = 35
