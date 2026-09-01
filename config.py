# config.py
# Handles loading of ML models and environment variables.
import os
import logging
import warnings
import functools
import spacy
from langchain_groq import ChatGroq
from dotenv import load_dotenv


logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*duckduckgo_search.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*torch\.jit\.script.*")
warnings.filterwarnings("ignore", message=r".*HF_TOKEN.*")

load_dotenv()

@functools.lru_cache(maxsize=1)
def load_spacy_model():
    try:
        import en_core_web_sm
        return en_core_web_sm.load()
    except ImportError:
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")

@functools.lru_cache(maxsize=1)
def load_nli_model():
    # Hardcoded bypass for cloud memory limits
    return None

def get_llm(model_alias: str = "A", max_tokens: int = None) -> ChatGroq:
    # Returns a ChatGroq LLM based on environment alias.
    model_env_key = f"GROQ_MODEL_{model_alias}"
    
    default_models = {
        "A": "qwen/qwen3.8-27b",
        "B": "qwen/qwen3.8-27b",
        "MEDIATOR": "qwen/qwen3.8-27b"
    }
    
    model_name = os.getenv(model_env_key, default_models.get(model_alias, "qwen/qwen3.8-27b"))
    kwargs = {"model": model_name, "temperature": 0.5, "timeout": 20}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return ChatGroq(**kwargs)
