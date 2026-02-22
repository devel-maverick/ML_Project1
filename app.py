import streamlit as st
import joblib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.preprocess import preprocess_pipeline
import requests
from bs4 import BeautifulSoup
import re

st.set_page_config(page_title="News Credibility Analyzer", page_icon="📰", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load("models/classifier.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer

def scrape_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title = ""
        text = ""
        
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

def predict_news(news_text, model, vectorizer):
    processed = preprocess_pipeline(news_text)
    features = vectorizer.transform([processed])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    confidence = max(probability)
    
    classification = "CREDIBLE" if prediction == 1 else "FAKE NEWS"
    credibility_score = probability[1] if len(probability) > 1 else confidence
    
    return classification, credibility_score, confidence

st.title("📰 Intelligent News Credibility Analyzer")
st.markdown("Analyze news articles for credibility and misinformation risk")

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input News Article")
    
    input_method = st.radio("Choose input method:", ["Paste Text", "Enter URL", "Upload File"], horizontal=True)
    
    news_text = ""
    url = ""
    title = ""
    
    if input_method == "Paste Text":
        news_text = st.text_area("Paste news article content:", height=250, placeholder="Enter the news article text here...")
        title = st.text_input("Article Title (optional):")
    
    elif input_method == "Enter URL":
        url = st.text_input("Enter article URL:", placeholder="https://example.com/news-article")
    
    elif input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload text file", type=["txt"])
        if uploaded_file:
            news_text = uploaded_file.read().decode("utf-8")
    
    analyze_btn = st.button("Analyze Credibility", type="primary", use_container_width=True)

with col2:
    st.subheader("Quick Stats")
    if "article_count" not in st.session_state:
        st.session_state.article_count = 0
        st.session_state.fake_count = 0
        st.session_state.credible_count = 0
    
    st.metric("Articles Analyzed", st.session_state.article_count)
    st.metric("Fake News Detected", st.session_state.fake_count)
    st.metric("Credible News", st.session_state.credible_count)

st.divider()

if analyze_btn:
    if input_method == "Paste Text" and not news_text.strip():
        st.warning("Please enter some text to analyze.")
    elif input_method == "Enter URL" and not url.strip():
        st.warning("Please enter a URL to analyze.")
    elif input_method == "Upload File" and not news_text.strip():
        st.warning("Please upload a file.")
    else:
        with st.spinner("Analyzing article..."):
            try:
                model, vectorizer = load_model()
                
                if input_method == "Enter URL":
                    scraped_title, scraped_text = scrape_url(url)
                    if not scraped_text:
                        st.error("Could not extract content from URL.")
                    else:
                        news_text = scraped_text
                        title = scraped_title
                        st.subheader("Scraped Content")
                        st.text_area(f"Title: {title}", news_text, height=200, disabled=True)
                
                if news_text:
                    classification, credibility_score, confidence = predict_news(news_text, model, vectorizer)
                    
                    st.session_state.article_count += 1
                    if classification == "CREDIBLE":
                        st.session_state.credible_count += 1
                    else:
                        st.session_state.fake_count += 1
                    
                    st.subheader("Analysis Results")
                    
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        st.metric("Credibility Score", f"{credibility_score:.2%}")
                    
                    with result_col2:
                        color = "normal" if classification == "CREDIBLE" else "inverse"
                        st.metric("Classification", classification)
                    
                    with result_col3:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    st.subheader("Interpretation")
                    if classification == "CREDIBLE":
                        st.success("This article appears to be **credible** based on textual features.")
                        st.info("Key indicators: Balanced language, verifiable content style, trustworthy structure")
                    else:
                        st.error("This article shows signs of **being fake news** or unreliable.")
                        st.warning("Risk factors: Sensationalist language, clickbait style, suspicious patterns detected")
                    
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

st.sidebar.header("About")
st.sidebar.info(
    "This tool analyzes news articles for credibility using "
    "machine learning and NLP techniques."
)

st.sidebar.header("Settings")
show_detailed = st.sidebar.checkbox("Show detailed analysis", value=False)
include_source = st.sidebar.checkbox("Include source analysis", value=True)
confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.1)
