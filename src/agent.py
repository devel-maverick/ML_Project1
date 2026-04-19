"""
Agent module - runs the full credibility analysis pipeline.
Steps: Classify -> Analyze -> Verify -> LLM -> Report
Each step updates a shared state dictionary.
"""

import time
from src.predict import predict_article, get_model_and_vectorizer
from src.explain import explain_prediction
from src.fact_checker import search_related_articles
from src.groq_client import get_credibility_assessment
from features.feature_engineering import get_sentiment_score, get_style_features


def create_initial_state(article_text, article_title=""):
    """Set up empty state dict that gets filled as pipeline runs."""
    return {
        "article_text": article_text,
        "article_title": article_title,
        "completed_steps": [],
        "current_step": None,
        "errors": [],
        "start_time": time.time(),
        "ml_prediction": None,
        "indicators": None,
        "verification": None,
        "llm_assessment": None,
        "final_report": None,
        "execution_time": None,
    }


# Step 1: Run ML model to classify article
def step_classify(state):
    state["current_step"] = "ML Classification"
    try:
        result = predict_article(state["article_text"])
        state["ml_prediction"] = result
        state["completed_steps"].append("classify")
    except Exception as e:
        state["errors"].append(f"Classification failed: {str(e)}")
    return state


# Step 2: Extract sentiment, style features, and top words
def step_analyze_indicators(state):
    state["current_step"] = "Indicator Analysis"
    try:
        sentiment = get_sentiment_score(state["article_text"])
        style = get_style_features(state["article_text"])
        model, vectorizer = get_model_and_vectorizer()
        top_words = explain_prediction(state["article_text"], model, vectorizer, top_n=10)

        state["indicators"] = {
            "sentiment": sentiment,
            "style_features": style,
            "top_words": top_words
        }
        state["completed_steps"].append("analyze")
    except Exception as e:
        # Use defaults if analysis fails
        state["indicators"] = {"sentiment": 0.0, "style_features": [0,0,0,0,0], "top_words": []}
        state["errors"].append(f"Indicator analysis failed: {str(e)}")
    return state


# Step 3: Search for related articles online
def step_verify_sources(state):
    state["current_step"] = "Cross-Source Verification"
    try:
        result = search_related_articles(state["article_text"])
        state["verification"] = result
        state["completed_steps"].append("verify")
    except Exception as e:
        state["verification"] = {"query": "", "articles": [], "status": "error",
                                  "message": f"Verification failed: {str(e)}"}
        state["errors"].append(f"Verification failed: {str(e)}")
    return state


# Step 4: Send everything to Groq LLM for analysis
def step_llm_reasoning(state):
    state["current_step"] = "LLM Reasoning"
    try:
        assessment = get_credibility_assessment(
            article_text=state["article_text"],
            ml_prediction=state["ml_prediction"],
            indicators=state["indicators"],
            verification_data=state["verification"]
        )
        state["llm_assessment"] = assessment
        if "error" not in assessment:
            state["completed_steps"].append("llm_reasoning")
        else:
            state["errors"].append(f"LLM note: {assessment['error']}")
    except Exception as e:
        state["llm_assessment"] = {
            "article_summary": "LLM analysis unavailable.",
            "credibility_indicators": [], "risk_factors": [],
            "cross_source_assessment": "Unavailable",
            "confidence_level": "N/A",
            "confidence_explanation": str(e),
            "supporting_sources": [],
            "recommendation": "Using ML-only results.",
            "disclaimer": "This is an automated analysis. Always verify with trusted sources.",
        }
        state["errors"].append(f"LLM failed: {str(e)}")
    return state


# Step 5: Put everything together into one report
def step_build_report(state):
    state["current_step"] = "Building Report"
    try:
        ml = state["ml_prediction"]
        llm = state["llm_assessment"]
        ind = state["indicators"]
        ver = state["verification"]

        label = "CREDIBLE" if ml and ml["prediction"] == 1 else "FAKE NEWS"

        state["final_report"] = {
            "classification": label,
            "ml_confidence": ml["confidence"] if ml else 0.0,
            "credibility_score": ml["probabilities"][1] if ml else 0.0,
            # LLM parts
            "article_summary": llm.get("article_summary", "No summary."),
            "credibility_indicators": llm.get("credibility_indicators", []),
            "risk_factors": llm.get("risk_factors", []),
            "cross_source_assessment": llm.get("cross_source_assessment", "N/A"),
            "confidence_level": llm.get("confidence_level", "N/A"),
            "confidence_explanation": llm.get("confidence_explanation", ""),
            "recommendation": llm.get("recommendation", ""),
            "disclaimer": llm.get("disclaimer", "Automated analysis. Verify with trusted sources."),
            # Indicators
            "sentiment": ind["sentiment"] if ind else 0.0,
            "style_features": ind["style_features"] if ind else [],
            "top_words": ind["top_words"] if ind else [],
            # Verification
            "verification_query": ver["query"] if ver else "",
            "verification_articles": ver["articles"] if ver else [],
            "verification_status": ver["status"] if ver else "unavailable",
            # Meta
            "errors": state["errors"],
            "completed_steps": state["completed_steps"],
            "article_title": state["article_title"],
        }
        state["completed_steps"].append("report")
    except Exception as e:
        state["errors"].append(f"Report build failed: {str(e)}")

    state["execution_time"] = round(time.time() - state["start_time"], 2)
    return state


def run_agent(article_text, article_title="", progress_callback=None):
    """
    Main entry point - runs all 5 steps of the pipeline.
    progress_callback is optional, used for UI updates.
    """
    state = create_initial_state(article_text, article_title)

    steps = [
        ("🔍 Step 1/5: Classifying with ML model...", step_classify),
        ("📊 Step 2/5: Analyzing indicators...", step_analyze_indicators),
        ("🌐 Step 3/5: Cross-referencing sources...", step_verify_sources),
        ("🤖 Step 4/5: AI reasoning...", step_llm_reasoning),
        ("📋 Step 5/5: Building report...", step_build_report),
    ]

    for step_name, step_fn in steps:
        if progress_callback:
            progress_callback(step_name)
        state = step_fn(state)

    return state
