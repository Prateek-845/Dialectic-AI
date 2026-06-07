from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import sys


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

from typing import Optional
import urllib.request
import re

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

from fastapi.responses import StreamingResponse
import json

@app.post("/analyze/stream")
async def analyze_article_stream(req: AnalyzeRequest):
    if not req.article:
        raise HTTPException(status_code=400, detail="Article text is required.")
        
    def generate():
        try:

            actual_article_text = req.article
            if actual_article_text.startswith("http://") or actual_article_text.startswith("https://"):
                actual_article_text = fetch_url_content(actual_article_text)
                
            graph = build_graph()
            config = {"configurable": {"thread_id": req.thread_id}}
            

            current_state = graph.get_state(config)
            
            if current_state.next:

                update_payload = {}
                if req.action:
                    update_payload["force_route"] = req.action
                if req.jury_feedback:
                    update_payload["jury_feedback"] = req.jury_feedback
                    
                if update_payload:
                    graph.update_state(config, update_payload)
                    

                for event in graph.stream(None, config=config, stream_mode="values"):
                    yield f"data: {json.dumps(event)}\n\n"
            else:

                for event in graph.stream({"original_article": actual_article_text}, config=config, stream_mode="values"):
                    yield f"data: {json.dumps(event)}\n\n"
                    
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
