import asyncio
from duckduckgo_search import DDGS

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

import re
from langchain_core.messages import HumanMessage

def extract_tag(content: str, tag: str) -> str:
    # Extracts text within XML-like tags, falling back to stripping <think>.
    match = re.search(f'<{tag}>(.*?)</{tag}>', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

async def generate_argument(llm, prompt: str, tag: str) -> str:
    # Helper to asynchronously run the LLM and extract the specified tag.
    result = await llm.ainvoke([HumanMessage(content=prompt)])
    return extract_tag(result.content.strip(), tag)
