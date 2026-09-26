import asyncio
import re
from duckduckgo_search import DDGS
from langchain_core.messages import HumanMessage

def _sync_web_search(query: str) -> str:
    # Performs a web search using the synchronous DDGS client.
    try:
        with DDGS(timeout=10) as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return "No external context found."
            context = []
            for r in results:
                context.append(f"Source: {r.get('title', 'Unknown')}\nSummary: {r.get('body', '')}")
            return "\n\n".join(context)
    except Exception as e:
        return f"No external context found. Error: {str(e)}"

async def perform_web_search(query: str) -> str:
    # Offloads the synchronous DuckDuckGo search to a separate worker thread.
    return await asyncio.to_thread(_sync_web_search, query)


async def safe_ainvoke(llm, messages, retries: int = 3, base_wait: float = 5.0):
    # Invokes the LLM with automatic retry on 429 rate-limit errors.
    # Waits base_wait * attempt seconds between retries (5s, 10s, 15s).
    for attempt in range(retries):
        try:
            return await llm.ainvoke(messages)
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait = base_wait * (attempt + 1)
                await asyncio.sleep(wait)
            else:
                raise


def extract_tag(content: str, tag: str) -> str:
    # Extracts text within XML-like tags, falling back to stripping <think>.
    match = re.search(f'<{tag}>(.*?)</{tag}>', content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    stripped = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
    return stripped if stripped else content.strip()

async def generate_argument(llm, prompt: str, tag: str) -> str:
    # Helper to asynchronously run the LLM and extract the specified tag.
    # Uses safe_ainvoke for automatic 429 retry with backoff.
    result = await safe_ainvoke(llm, [HumanMessage(content=prompt)])
    return extract_tag(result.content.strip(), tag)
