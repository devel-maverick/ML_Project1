"""
Groq LLM Client - handles communication with the Groq API
for generating credibility assessment reports.
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# System prompt that tells the LLM how to behave and respond
SYSTEM_PROMPT = """You are a misinformation analyst. Analyze news articles for credibility 
based on the evidence provided.

Rules:
- Only use the evidence given to you (ML results, indicators, sources)
- Don't make up sources that weren't provided
- Be honest if evidence is insufficient
- Include a disclaimer about automated analysis limitations

Respond in this exact JSON format (no markdown, no code blocks):
{
    "article_summary": "2-3 sentence summary of the article",
    "credibility_indicators": ["list of positive credibility signals"],
    "risk_factors": ["list of red flags found"],
    "cross_source_assessment": "how claims compare with other sources",
    "confidence_level": "HIGH / MEDIUM / LOW",
    "confidence_explanation": "why this confidence level",
    "supporting_sources": ["relevant sources from evidence"],
    "recommendation": "clear recommendation about credibility",
    "disclaimer": "limitations disclaimer"
}"""


def get_credibility_assessment(article_text, ml_prediction, indicators, verification_data):
    """
    Send all gathered evidence to Groq LLM and get a structured assessment.
    Loads API key from .env file (backend only, not user-facing).
    """
    # Check if API key is configured
    if not GROQ_API_KEY or GROQ_API_KEY == "your_api_key_here":
        return {
            "article_summary": "LLM analysis unavailable - API key not configured.",
            "credibility_indicators": [],
            "risk_factors": [],
            "cross_source_assessment": "Unavailable",
            "confidence_level": "N/A",
            "confidence_explanation": "Configure GROQ_API_KEY in .env file.",
            "supporting_sources": [],
            "recommendation": "Using ML-only analysis.",
            "disclaimer": "This is an automated analysis. Always verify with trusted sources.",
            "error": "API key not set"
        }

    try:
        client = Groq(api_key=GROQ_API_KEY)

        # Truncate article to keep within token limits
        short_text = article_text[:3000]
        if len(article_text) > 3000:
            short_text += "..."

        # Format the ML prediction info
        label = "CREDIBLE" if ml_prediction["prediction"] == 1 else "FAKE NEWS"
        confidence = f"{ml_prediction['confidence']:.2%}"

        # Format verification sources
        if verification_data["status"] == "success" and verification_data["articles"]:
            sources = "\n".join([
                f"  - \"{a['title']}\" ({a['source']})"
                for a in verification_data["articles"]
            ])
            verify_text = f"Related articles found:\n{sources}"
        else:
            verify_text = verification_data["message"]

        # Build the prompt with all evidence
        prompt = f"""Analyze this news article:

ARTICLE:
{short_text}

ML MODEL RESULTS:
- Classification: {label}
- Confidence: {confidence}
- P(Real): {ml_prediction['probabilities'][1]:.4f}
- P(Fake): {ml_prediction['probabilities'][0]:.4f}

INDICATORS:
- Sentiment: {indicators['sentiment']:.3f}
- Exclamations: {int(indicators['style_features'][0])}
- Questions: {int(indicators['style_features'][1])}
- Caps ratio: {indicators['style_features'][2]*100:.1f}%
- Avg sentence length: {indicators['style_features'][3]:.1f}
- Word count: {int(indicators['style_features'][4])}

TOP WORDS: {', '.join([f'{w} ({s:.4f})' for w, s in indicators['top_words'][:10]])}

VERIFICATION:
{verify_text}

Give your structured JSON assessment based on ALL evidence above."""

        # Call the Groq API
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1500,
        )

        result = response.choices[0].message.content.strip()

        # Clean up markdown wrappers if present
        if result.startswith("```"):
            result = result.split("\n", 1)[1]
        if result.endswith("```"):
            result = result.rsplit("```", 1)[0]

        return json.loads(result)

    except json.JSONDecodeError:
        return {
            "article_summary": "Could not parse LLM response properly.",
            "credibility_indicators": [],
            "risk_factors": ["Response format error"],
            "cross_source_assessment": result if 'result' in dir() else "Unavailable",
            "confidence_level": "LOW",
            "confidence_explanation": "Response parsing failed.",
            "supporting_sources": [],
            "recommendation": "Review the raw analysis manually.",
            "disclaimer": "This is an automated analysis. Always verify with trusted sources.",
            "error": "JSON parsing failed"
        }
    except Exception as e:
        return {
            "article_summary": "LLM analysis failed.",
            "credibility_indicators": [],
            "risk_factors": [],
            "cross_source_assessment": "Unavailable",
            "confidence_level": "N/A",
            "confidence_explanation": str(e),
            "supporting_sources": [],
            "recommendation": "Falling back to ML-only results.",
            "disclaimer": "This is an automated analysis. Always verify with trusted sources.",
            "error": str(e)
        }
