
from setup_nltk import download_nltk_resources
download_nltk_resources()

import os
import re
import time
import traceback
from pathlib import Path

import numpy as np
import streamlit as st
import requests
from bs4 import BeautifulSoup

from src.predict import predict_article, get_model_and_vectorizer
from src.explain import explain_prediction
from features.feature_engineering import get_sentiment_score, get_style_features
from src.agent import run_agent
from src.pdf_report import generate_pdf_report


# Page config
st.set_page_config(page_title="News Credibility Analyzer", page_icon="📰", layout="wide")


def scrape_url(url):
    """Scrape article title and text from a URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        title = ""
        title_tags = soup.find_all(['h1', 'title', 'h2'])
        if title_tags:
            title = title_tags[0].get_text().strip()

        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        text = re.sub(r'\s+', ' ', text).strip()

        return title, text
    except Exception as e:
        st.error(f"Error scraping URL: {str(e)}")
        return "", ""


# --- Sidebar: Session Stats ---
with st.sidebar:
    st.header("📊 Session Stats")
    if "article_count" not in st.session_state:
        st.session_state.article_count = 0
        st.session_state.fake_count = 0
        st.session_state.credible_count = 0

    st.metric("Articles Analyzed", st.session_state.article_count)
    st.metric("Fake News Detected", st.session_state.fake_count)
    st.metric("Credible News", st.session_state.credible_count)


# --- Main Title ---
st.title("📰 Intelligent News Credibility Analyzer")
st.markdown("Analyze news articles for credibility and misinformation risk")
st.divider()


# --- Input Section (shared by both tabs) ---
st.subheader("Input News Article")
input_method = st.radio("Choose input method:", ["Paste Text", "Enter URL", "Upload File"], horizontal=True)

news_text = ""
url = ""
title = ""

if input_method == "Paste Text":
    news_text = st.text_area("Paste news article content:", height=200,
                              placeholder="Enter the news article text here...")
    title = st.text_input("Article Title (optional):")
elif input_method == "Enter URL":
    url = st.text_input("Enter article URL:", placeholder="https://example.com/news-article")
elif input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload text file", type=["txt"])
    if uploaded_file:
        try:
            news_text = uploaded_file.read().decode("utf-8")
            st.caption(f"📄 {uploaded_file.name} — {len(news_text):,} characters loaded")
        except UnicodeDecodeError:
            st.error("Could not read file: please upload a UTF-8 encoded .txt file.")

st.divider()


# --- Three Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Quick Analysis", "🤖 AI Agent Analysis", "📈 Model Evaluation"])


with tab1:
    st.markdown("*Fast ML-based classification — no API key needed*")
    analyze_btn = st.button("⚡ Quick Analyze", type="primary", use_container_width=True, key="quick")

    if analyze_btn:
        if input_method == "Paste Text" and not news_text.strip():
            st.warning("Please enter some text.")
        elif input_method == "Enter URL" and not url.strip():
            st.warning("Please enter a URL.")
        elif input_method == "Upload File" and not news_text.strip():
            st.warning("Please upload a file.")
        else:
            # Handle URL scraping before spinner so content stays visible
            if input_method == "Enter URL":
                with st.spinner("Scraping article…"):
                    scraped_title, scraped_text = scrape_url(url)
                if not scraped_text:
                    st.error("Could not extract content from URL.")
                    st.stop()
                news_text = scraped_text
                title = scraped_title
                with st.expander(f"📄 Scraped: {title}", expanded=False):
                    st.write(news_text[:2000] + ("…" if len(news_text) > 2000 else ""))

            with st.spinner("Analyzing..."):
                try:
                    if news_text:
                        result = predict_article(news_text)
                        prediction = result["prediction"]
                        confidence = result["confidence"]
                        probabilities = result["probabilities"]
                        classification = "CREDIBLE" if prediction == 1 else "FAKE NEWS"
                        credibility_score = probabilities[1]

                        # Update stats
                        st.session_state.article_count += 1
                        if classification == "CREDIBLE":
                            st.session_state.credible_count += 1
                        else:
                            st.session_state.fake_count += 1

                        # Fetch model once for explain + session state
                        model, vectorizer = get_model_and_vectorizer()
                        sentiment = get_sentiment_score(news_text)
                        style_features = get_style_features(news_text)
                        contributions = explain_prediction(news_text, model, vectorizer, top_n=10)

                        # Persist result in session state so tab-switching doesn't clear it
                        st.session_state["quick_result"] = {
                            "credibility_score": credibility_score,
                            "classification": classification,
                            "confidence": confidence,
                            "sentiment": sentiment,
                            "style_features": style_features,
                            "contributions": contributions,
                        }

                        # Show results
                        st.subheader("Analysis Results")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Credibility Score", f"{credibility_score:.2%}")
                        c2.metric("Classification", classification)
                        c3.metric("Confidence", f"{confidence:.2%}")

                        st.divider()

                        # Patterns
                        st.subheader("Detected Patterns")
                        p1, p2, p3 = st.columns(3)
                        with p1:
                            delta_col = "normal" if sentiment >= 0 else "inverse"
                            st.metric("Sentiment", f"{sentiment:.2f}",
                                      delta=f"{'Positive' if sentiment >= 0 else 'Negative'}",
                                      delta_color=delta_col)
                        with p2:
                            st.metric("Exclamations", int(style_features[0]))
                            st.metric("Questions", int(style_features[1]))
                        with p3:
                            st.metric("Caps Ratio", f"{style_features[2]*100:.1f}%")
                            st.metric("Avg Sentence Len", f"{style_features[3]:.1f}")

                        st.divider()

                        # Top words
                        st.subheader("Key Influential Features")

                        if contributions:
                            w1, w2 = st.columns(2)
                            with w1:
                                st.write("**Indicates Real News:**")
                                for word, score in contributions:
                                    if score > 0:
                                        st.markdown(f"<span style='color:green'>• {word}</span> ({score:.4f})",
                                                    unsafe_allow_html=True)
                            with w2:
                                st.write("**Indicates Fake News:**")
                                for word, score in contributions:
                                    if score < 0:
                                        st.markdown(f"<span style='color:red'>• {word}</span> ({score:.4f})",
                                                    unsafe_allow_html=True)

                        st.divider()
                        st.subheader("Interpretation")
                        if classification == "CREDIBLE":
                            st.success("This article appears to be **credible**.")
                        else:
                            st.error("This article shows signs of **fake news**.")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.error(traceback.format_exc())

    # Show last result if user switched tabs and came back
    elif "quick_result" in st.session_state:
        r = st.session_state["quick_result"]
        st.info("Showing last analysis result. Submit again to refresh.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Credibility Score", f"{r['credibility_score']:.2%}")
        c2.metric("Classification", r["classification"])
        c3.metric("Confidence", f"{r['confidence']:.2%}")

with tab2:
    st.markdown("*Full AI analysis with LLM reasoning, source verification, and PDF export*")
    agent_btn = st.button("🤖 Run AI Agent Analysis", type="primary",
                           use_container_width=True, key="agent")

    if agent_btn:
        if input_method == "Paste Text" and not news_text.strip():
            st.warning("Please enter some text.")
        elif input_method == "Enter URL" and not url.strip():
            st.warning("Please enter a URL.")
        elif input_method == "Upload File" and not news_text.strip():
            st.warning("Please upload a file.")
        else:
            # Handle URL
            if input_method == "Enter URL":
                with st.spinner("Scraping article..."):
                    scraped_title, scraped_text = scrape_url(url)
                    if not scraped_text:
                        st.error("Could not extract content.")
                        st.stop()
                    news_text = scraped_text
                    title = scraped_title

            # Run agent pipeline with progress
            st.subheader("🤖 Agent Pipeline")
            progress = st.progress(0)
            status = st.empty()

            counter = {"n": 0}
            def on_progress(step_name):
                counter["n"] += 1
                progress.progress(counter["n"] / 5)
                status.info(step_name)

            state = run_agent(
                article_text=news_text,
                article_title=title,
                progress_callback=on_progress
            )

            progress.progress(1.0)
            status.success(f"✅ Done in {state['execution_time']:.1f}s")

            report = state.get("final_report", {})

            # Update stats
            st.session_state.article_count += 1
            if report.get("classification") == "CREDIBLE":
                st.session_state.credible_count += 1
            else:
                st.session_state.fake_count += 1

            # --- Results Display ---
            st.divider()
            st.subheader("📋 Credibility Report")

            classification = report.get("classification", "UNKNOWN")

            # Top metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Classification", classification)
            m2.metric("Credibility", f"{report.get('credibility_score', 0):.2%}")
            m3.metric("ML Confidence", f"{report.get('ml_confidence', 0):.2%}")
            m4.metric("AI Confidence", report.get("confidence_level", "N/A"))

            st.divider()

            # Article summary
            with st.expander("📝 Article Summary", expanded=True):
                st.write(report.get("article_summary", "No summary."))

            # Indicators and risks side by side
            col_a, col_b = st.columns(2)
            with col_a:
                with st.expander("✅ Credibility Indicators", expanded=True):
                    for item in report.get("credibility_indicators", []):
                        st.markdown(f"<span style='color:green'>✓ {item}</span>", unsafe_allow_html=True)
                    if not report.get("credibility_indicators"):
                        st.info("None identified.")

            with col_b:
                with st.expander("⚠️ Risk Factors", expanded=True):
                    for item in report.get("risk_factors", []):
                        st.markdown(f"<span style='color:red'>⚡ {item}</span>", unsafe_allow_html=True)
                    if not report.get("risk_factors"):
                        st.info("None detected.")

            # Cross-source verification
            with st.expander("🌐 Cross-Source Verification", expanded=True):
                st.write(report.get("cross_source_assessment", "Unavailable."))
                articles = report.get("verification_articles", [])
                if articles:
                    st.markdown("**Related Sources:**")
                    for a in articles:
                        st.markdown(f"- [{a['title']}]({a['url']}) — *{a['source']}*")
                else:
                    st.info("No related articles found.")

            # Content metrics
            with st.expander("📊 Content Metrics", expanded=False):
                sentiment = report.get("sentiment", 0)
                style = report.get("style_features", [0,0,0,0,0])
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.metric("Sentiment", f"{sentiment:.3f}",
                              delta="Positive" if sentiment >= 0 else "Negative",
                              delta_color="normal" if sentiment >= 0 else "inverse")
                with s2:
                    st.metric("Exclamations", int(style[0]))
                    st.metric("Questions", int(style[1]))
                with s3:
                    st.metric("Caps Ratio", f"{style[2]*100:.1f}%")
                    st.metric("Avg Sentence", f"{style[3]:.1f}")

            # Confidence & recommendation
            with st.expander("🎯 Confidence & Recommendation", expanded=True):
                st.write(f"**Level:** {report.get('confidence_level', 'N/A')}")
                st.write(report.get("confidence_explanation", ""))
                st.divider()
                st.write(f"**Recommendation:** {report.get('recommendation', 'N/A')}")

            # Disclaimer
            with st.expander("⚖️ Disclaimer", expanded=False):
                st.warning(report.get("disclaimer", "Automated analysis. Verify with trusted sources."))

            # Errors/notes
            errors = report.get("errors", [])
            if errors:
                with st.expander("⚠️ Notes", expanded=False):
                    for e in errors:
                        st.caption(f"• {e}")

            # --- PDF Download ---
            st.divider()
            st.subheader("📥 Export Report")
            try:
                pdf_bytes = generate_pdf_report(report)
                fname = f"credibility_report_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
                st.download_button("📄 Download PDF Report", data=pdf_bytes,
                                    file_name=fname, mime="application/pdf",
                                    use_container_width=True)
            except Exception as e:
                st.error(f"PDF error: {str(e)}")

with tab3:
    st.markdown("*Evaluation visualisations generated by `src/train.py`.*")
    REPORTS_DIR = Path(__file__).resolve().parent / "reports"

    cm_path = REPORTS_DIR / "confusion_matrix.png"
    roc_path = REPORTS_DIR / "roc_curve.png"

    if not cm_path.exists() and not roc_path.exists():
        st.info(
            "No evaluation plots found. Run `python src/train.py` once to generate "
            "`reports/confusion_matrix.png` and `reports/roc_curve.png`."
        )
    else:
        col_left, col_right = st.columns(2)
        with col_left:
            if cm_path.exists():
                st.subheader("Confusion Matrix")
                st.image(str(cm_path))
            else:
                st.warning("confusion_matrix.png not found.")
        with col_right:
            if roc_path.exists():
                st.subheader("ROC Curve")
                st.image(str(roc_path))
            else:
                st.warning("roc_curve.png not found.")
