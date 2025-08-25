import os
import logging
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class _NoLLM:
    """Graceful fallback when no API key is present; .invoke() will raise,
    and callers should already have try/except fallbacks."""
    def invoke(self, *_, **__):
        raise RuntimeError("LLM unavailable: missing or invalid OpenAI configuration.")

def build_llm():
    """Build a ChatOpenAI from env (MODEL_NAME, MODEL_TEMPERATURE).
    Falls back to _NoLLM if OPENAI_API_KEY is missing."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("MODEL_NAME", "gpt-4").strip()
    try:
        temperature = float(os.getenv("MODEL_TEMPERATURE", "0"))
    except ValueError:
        temperature = 0.0

    if not api_key:
        logger.warning("OPENAI_API_KEY not set; LLM calls will be skipped with graceful fallbacks.")
        return _NoLLM()

    # Single place to configure the LLM client
    return ChatOpenAI(model=model, temperature=temperature)