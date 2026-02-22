import pandas as pd
import re
import string
import nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def load_data(file_path):
    df=pd.read_csv(file_path ,encoding="utf-8")
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    before_empty = len(df)
    df=df[df['content'].str.strip()!='']
    after_empty = len(df)
    print(f"Removed {before_empty-after_empty} rows with empty content.")
    before = len(df)
    df = df.drop_duplicates(subset="content")
    after = len(df)
    print(f"Removed {before-after} duplicate rows.")
    df = df.reset_index(drop=True)
    return df

def preprocess_text(text):
    text=text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    text=re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def tokenize(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess_pipeline(text):
    text = preprocess_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


def preprocess_entire_dataframe(df):

    print("\nApplying NLP preprocessing...")
    tqdm.pandas()
    df['processed_content'] = df['content'].progress_apply(preprocess_pipeline)
    return df

def tfidf_vectorizer(df):
    print("\nCreating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000,ngram_range=(1,2),min_df=5)
    matrix = vectorizer.fit_transform(df['processed_content'])
    print("matrix shape:", matrix.shape)
    return matrix,vectorizer
