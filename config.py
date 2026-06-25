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
    print("--- [Config] Loading spaCy model 'en_core_web_sm' ---")
    try:
        import en_core_web_sm
        model = en_core_web_sm.load()
        print("--- [Config] Loaded spaCy model via direct package import ---")
        return model
    except ImportError:
        print("--- [Config] Direct import of en_core_web_sm failed. Trying spacy.load ---")
        try:
            model = spacy.load("en_core_web_sm")
            print("--- [Config] Loaded spaCy model via spacy.load ---")
            return model
        except OSError as e:
            print(f"--- [Config] spacy.load failed: {str(e)}. Attempting CLI download fallback ---")
            from spacy.cli import download
            download("en_core_web_sm")
            model = spacy.load("en_core_web_sm")
            print("--- [Config] Loaded spaCy model after fallback CLI download ---")
            return model

@functools.lru_cache(maxsize=1)
def load_nli_model():
    # Hardcoded bypass for cloud memory limits
    return None

def get_llm(model_alias: str = "A", max_tokens: int = None) -> ChatGroq:
    # Returns a ChatGroq LLM based on environment alias.
    model_env_key = f"GROQ_MODEL_{model_alias}"
    model_name = os.getenv(model_env_key, "llama-3.3-70b-versatile")
    kwargs = {"model": model_name, "temperature": 0.5, "timeout": 20}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return ChatGroq(**kwargs)
