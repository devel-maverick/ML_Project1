# 📰 Intelligent News Credibility Analysis and Agentic Misinformation Monitoring System

## Milestone 1 -- ML-Based News Credibility Classification

------------------------------------------------------------------------

## 📌 Project Overview

This project presents an AI-driven system for analyzing the credibility
of news articles using Natural Language Processing (NLP) and supervised
machine learning techniques.

The system classifies news content as:

-   ✅ **Credible (Real News)**
-   ❌ **Fake / Misinformation**

The application integrates: - Text preprocessing (cleaning,
tokenization, lemmatization) - TF-IDF feature extraction - Logistic
Regression classification - Linguistic pattern analysis - Explainable AI
(feature contribution analysis) - Interactive Streamlit-based user
interface

The application currently supports: - Text input - URL scraping  - File upload(.txt files)

------------------------------------------------------------------------

## 🎯 Objectives

-   Build a supervised ML model to detect misinformation.
-   Analyze linguistic and stylistic patterns in news content.
-   Provide explainable predictions.
-   Deploy a publicly accessible web interface.

------------------------------------------------------------------------

## 📊 Dataset

**WELFake Dataset (Kaggle)**\
- \~63,000 news articles\
- Binary labels: - `0` → Fake News\
- `1` → Real News

Dataset Link:\
https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

------------------------------------------------------------------------

## 🏗 System Architecture

    User Input (Text / URL / File)
                ↓
    Text Preprocessing
                ↓
    TF-IDF Feature Extraction
                ↓
    Logistic Regression Model
                ↓
    Prediction + Confidence Score
                ↓
    Explainability + Pattern Analysis
                ↓
    Streamlit Web Interface

------------------------------------------------------------------------

## 🧠 Machine Learning Details

### Model Used

-   Logistic Regression

### Feature Extraction

-   TF-IDF (Unigrams + Bigrams)
-   Max features: 5000

### Evaluation Metrics (Weighted Averages)

  Metric      Score
  ----------- -------
  Accuracy    0.95
  Precision   0.95
  Recall      0.95
  F1 Score    0.95

The model demonstrates strong generalization with balanced performance
across both classes.

------------------------------------------------------------------------

## 🚀 Tech Stack

-   Python 3.12
-   scikit-learn
-   pandas
-   numpy
-   NLTK
-   Streamlit
-   BeautifulSoup
-   Joblib

------------------------------------------------------------------------

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

``` bash
git clone https://github.com/YourUsername/YourRepositoryName.git
cd YourRepositoryName
```

### 2️⃣ Create Virtual Environment

``` bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3️⃣ Install Dependencies

``` bash
pip install -r requirements.txt
```

### 4️⃣ Download NLTK Data

``` bash
python setup_nltk.py
```

> **Note:** NLTK downloads `punkt`, `punkt_tab`, `stopwords`, and `wordnet` to
> `~/nltk_data` by default.  To use a custom path set the `NLTK_DATA`
> environment variable:
>
> ```bash
> export NLTK_DATA=/path/to/nltk_data
> ```

### 5️⃣ (Optional) Train the Model

``` bash
python src/train.py
```

This re-trains the classifier and saves evaluation plots to `reports/`.

### 6️⃣ Run Application

``` bash
streamlit run app.py
```

### 7️⃣ Run Unit Tests

``` bash
pytest tests/ -v
```

------------------------------------------------------------------------

## 🌍 Deployment

The project is deployed using **Streamlit Community Cloud**.

Public URL:\
https://news-credibility-analyzer.streamlit.app/
------------------------------------------------------------------------

## 📁 Project Structure

    .
    ├── app.py
    ├── config.py
    ├── setup_nltk.py
    ├── requirements.txt
    │
    ├── data/
    ├── models/
    │   ├── classifier.pkl
    │   └── vectorizer.pkl
    │
    ├── preprocessing/
    ├── features/
    ├── src/
    │   ├── train.py
    │   ├── predict.py
    │   ├── explain.py
    │
    └── README.md

------------------------------------------------------------------------

## 👥 Individual Contributions

### 🔹 Siddharth Kumar Shukla (Project Lead) -- Model Training & Evaluation & Report Creation

-   Implemented Logistic Regression classifier.
-   Performed train-test split and evaluation.
-   Computed accuracy, precision, recall, F1-score.
-   Saved trained model and vectorizer for deployment.

### 🔹 Vinay Sharma -- Data Preprocessing & NLP Pipeline

-   Implemented text cleaning, tokenization, stopword removal, and
    lemmatization.
-   Developed preprocessing pipeline.
-   Integrated TF-IDF feature extraction.

### 🔹 Sumit Kumar -- UI Development & Deployment

-   Developed Streamlit-based user interface.
-   Integrated prediction pipeline into UI.
-   Implemented URL scraping functionality.
-   Managed deployment on Streamlit Cloud.

### 🔹 Vansh Jain -- Explainability & Linguistic Analysis

-   Implemented feature contribution analysis.
-   Designed sentiment and stylistic feature extraction.
-   Added interpretability module to highlight influential words.
-   Conducted qualitative error analysis.

------------------------------------------------------------------------

## ⚠ Limitations

-   The model detects linguistic patterns rather than factual truth
    verification.
-   Performance depends on similarity to training dataset distribution.
-   Real-world misinformation detection may require external
    fact-checking systems.

------------------------------------------------------------------------

## 📌 Future Work (Milestone 2)

-   Agentic AI assistant for structured credibility reports
-   Real-time misinformation monitoring
-   Source reliability scoring
-   Embedding-based semantic analysis

------------------------------------------------------------------------
