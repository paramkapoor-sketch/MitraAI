"""Real-time news analysis with LLM

A Python module for analyzing news using various analytical styles.
"""

import os
from typing import Optional
from langchain_groq import ChatGroq
from ddgs import DDGS
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize the LLM
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is required")

llm_groq = ChatGroq(model_name="openai/gpt-oss-20b", api_key=api_key)


def news_analyzer(style: str, query: str, region: str = 'us-en') -> str:
    """Analyze news articles using a specified style.
    
    Args:
        style: The analytical style to apply
        query: The search query for news
        region: The region for news search (default: 'us-en')
        
    Returns:
        Analysis of the news articles in the specified style
    """
    try:
        text = ""
        ddgs = DDGS()
        articles = ddgs.news(query, region=region)
        
        for article in articles:
            title = article.get('title', '')
            body = article.get('body', '')
            text += f"{title}\n{body}\n\n"
        
        if not text.strip():
            return "No news articles found for the given query."
        
        prompt = (f"Give a detailed news analysis in this style: {style}. "
                 f"You will be given news items to analyze and apply that style. "
                 f"Here is the user question: {query}\n\n"
                 f"The news items are: {text}")
        
        response = llm_groq.invoke(prompt)
        return response.content
        
    except Exception as e:
        return f"Error analyzing news: {str(e)}"


# API Models
class NewsAnalysisRequest(BaseModel):
    style: str
    query: str
    region: Optional[str] = 'us-en'


class NewsAnalysisResponse(BaseModel):
    analysis: str
    query: str
    style: str


# FastAPI app
app = FastAPI(title="News Analysis API", description="Analyze news with various styles")


@app.post("/analyze", response_model=NewsAnalysisResponse)
async def analyze_news(request: NewsAnalysisRequest):
    """Analyze news articles with a specified style."""
    try:
        analysis = news_analyzer(request.style, request.query, request.region)
        return NewsAnalysisResponse(
            analysis=analysis,
            query=request.query,
            style=request.style
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)