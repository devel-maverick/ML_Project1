
from setup_nltk import download_nltk_resources
download_nltk_resources()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from src.agent import run_agent

app = FastAPI(title="News Credibility API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    article_text: str
    article_title: Optional[str] = ""


class AnalyzeResponse(BaseModel):
    classification: str
    ml_confidence: float
    credibility_score: float
    article_summary: str
    credibility_indicators: list
    risk_factors: list
    cross_source_assessment: str
    confidence_level: str
    recommendation: str
    disclaimer: str
    verification_articles: list
    errors: list
    execution_time: float


@app.get("/")
def health_check():
    return {"name": "News Credibility API", "version": "2.0.0", "status": "running"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_article(request: AnalyzeRequest):
    state = run_agent(
        article_text=request.article_text,
        article_title=request.article_title
    )
    report = state.get("final_report", {})

    return AnalyzeResponse(
        classification=report.get("classification", "UNKNOWN"),
        ml_confidence=report.get("ml_confidence", 0.0),
        credibility_score=report.get("credibility_score", 0.0),
        article_summary=report.get("article_summary", ""),
        credibility_indicators=report.get("credibility_indicators", []),
        risk_factors=report.get("risk_factors", []),
        cross_source_assessment=report.get("cross_source_assessment", ""),
        confidence_level=report.get("confidence_level", ""),
        recommendation=report.get("recommendation", ""),
        disclaimer=report.get("disclaimer", ""),
        verification_articles=report.get("verification_articles", []),
        errors=report.get("errors", []),
        execution_time=state.get("execution_time", 0.0)
    )
