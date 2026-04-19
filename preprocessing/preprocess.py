"""preprocess.py — NLP preprocessing pipeline for the WELFake news dataset.

Provides functions to clean raw text, tokenise, remove stop-words, lemmatise,
and run the full pipeline.  Also exposes a helper to create TF-IDF features.
"""

import pandas as pd
import re
import string
import nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("punkt_tab")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def load_data(file_path):
    """Load the WELFake CSV, combine title + text into a *content* column,
    and drop empty / duplicate rows.

    Parameters
    ----------
    file_path : str or Path
        Path to the WELFake CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with a *content* column.
    """
    df = pd.read_csv(file_path ,encoding = "utf-8")
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    before_empty = len(df)
    df=df[df["content"].str.strip()!=""]
    after_empty = len(df)
    print(f"Removed {before_empty-after_empty} rows with empty content.")
    before = len(df)
    df = df.drop_duplicates(subset="content")
    after = len(df)
    print(f"Removed {before-after} duplicate rows.")
    df = df.reset_index(drop=True)
    return df

def preprocess_text(text):
    """Apply regex-based cleaning to *text*.

    Removes URLs, e-mail addresses, punctuation, digits, and collapses
    consecutive whitespace.  Returns lower-cased result.
    """
    text=text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    text=re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def tokenize(text):
    """Tokenise *text* into a list of word tokens using NLTK's word_tokenize."""
    return word_tokenize(text)

def remove_stopwords(tokens):
    """Return *tokens* with English stop-words removed."""
    return [word for word in tokens if word not in stop_words]

def lemmatize_tokens(tokens):
    """Lemmatise each token in *tokens* using WordNetLemmatizer."""
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess_pipeline(text):
    """Run the full NLP pipeline on a single string.

    Steps: clean → tokenise → remove stop-words → lemmatise → join.

    Returns
    -------
    str
        Space-joined processed tokens.
    """
    text = preprocess_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


def preprocess_entire_dataframe(df):
    """Apply :func:`preprocess_pipeline` to every row in *df*.

    Adds a *processed_content* column and returns the modified dataframe.
    """
    print("\nApplying NLP preprocessing...")
    df["processed_content"] = df["content"].apply(preprocess_pipeline)
    return df

def tfidf_vectorizer(df):
    """Fit a TF-IDF vectorizer on *df['processed_content']* and transform it.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a *processed_content* column.

    Returns
    -------
    tuple[scipy.sparse.csr_matrix, TfidfVectorizer]
        The feature matrix and the fitted vectorizer.
    """
    print("\nCreating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000,ngram_range=(1,2),min_df=5)
    matrix = vectorizer.fit_transform(df["processed_content"])
    print("matrix shape:", matrix.shape)
    return matrix,vectorizer
