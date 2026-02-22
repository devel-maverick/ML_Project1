import numpy as np
from textblob import TextBlob


# ==============================
# 1️⃣ Sentiment Feature
# ==============================
def get_sentiment_score(text):
    """
    Returns sentiment polarity score:
    Range: -1 (negative) to +1 (positive)
    """
    return TextBlob(text).sentiment.polarity


# ==============================
# 2️⃣ Stylometric Features
# ==============================
def get_style_features(text):
    """
    Returns basic writing style features:
    - Exclamation count
    - Question mark count
    - Capital letter ratio
    - Average sentence length
    - Total word count
    """


    exclamation_count = text.count("!")
    question_count = text.count("?")

   
    capital_ratio = sum(1 for c in text if c.isupper()) / (len(text) + 1)

   
    sentences = text.split(".")
    sentence_count = len(sentences)
    word_count = len(text.split())

    avg_sentence_length = word_count / (sentence_count + 1)

    return np.array([
        exclamation_count,
        question_count,
        capital_ratio,
        avg_sentence_length,
        word_count
    ])


# ==============================
# 3️⃣ Combine All Custom Features
# ==============================
def extract_custom_features(text):
    """
    Combines sentiment + style features
    Returns numpy array
    """

    sentiment = get_sentiment_score(text)
    style = get_style_features(text)

    return np.hstack(([sentiment], style))