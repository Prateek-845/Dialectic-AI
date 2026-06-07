from duckduckgo_search import DDGS

def perform_web_search(query: str) -> str:
    """Performs a web search using DuckDuckGo and returns the results as a string."""
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No external context found."
        
        context = []
        for r in results:
            context.append(f"Source: {r.get('title', 'Unknown')}\nSummary: {r.get('body', '')}")
            
        return "\n\n".join(context)
    except Exception as e:
        return f"No external context found. Error: {str(e)}"
