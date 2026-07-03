from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from typing import Optional
import urllib.request
import re
import json
import asyncio
import uvicorn
import os
import sys

# Add the parent directory to Python path so we can import 'graph'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph import build_graph

app = FastAPI(title="Dialectic AI Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    article: str
    thread_id: str
    action: Optional[str] = None
    jury_feedback: Optional[str] = None

def fetch_url_content(url: str) -> str:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html_bytes = urllib.request.urlopen(req, timeout=10).read()
        html_str = html_bytes.decode('utf-8', errors='ignore')
        
        html_str = re.sub(r'<script.*?</script>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
        html_str = re.sub(r'<style.*?</style>', '', html_str, flags=re.DOTALL | re.IGNORECASE)

        text = re.sub(r'<[^>]+>', ' ', html_str)

        lines = (line.strip() for line in text.splitlines())
        text = '\n'.join(line for line in lines if line)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text[:3000]
        
    except Exception as e:
        return url


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/analyze/stream")
async def analyze_article_stream(req: AnalyzeRequest):
    try:
        graph = build_graph()
        config = {"configurable": {"thread_id": req.thread_id}}
        current_state = graph.get_state(config)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Graph initialization failed: {str(e)}")

    if not current_state.next and not req.article:
        raise HTTPException(status_code=400, detail="Article text is required to start a new analysis.")
        
    async def generate():
        import traceback
        try:
            actual_article = req.article
            if actual_article and actual_article.startswith(("http://", "https://")):
                actual_article = await asyncio.to_thread(fetch_url_content, actual_article)
                
            if current_state.next:
                update_payload = {k: v for k, v in [("force_route", req.action), ("jury_feedback", req.jury_feedback)] if v}
                if update_payload:
                    graph.update_state(config, update_payload)
                async for event in graph.astream(None, config=config, stream_mode="values"):
                    yield f"data: {json.dumps(event)}\n\n"
            else:
                async for event in graph.astream({"original_article": actual_article}, config=config, stream_mode="values"):
                    yield f"data: {json.dumps(event)}\n\n"
                    
            yield "data: [DONE]\n\n"
        except asyncio.CancelledError:
            pass
        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
