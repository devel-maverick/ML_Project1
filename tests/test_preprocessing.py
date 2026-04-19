"""Unit tests for preprocessing/preprocess.py."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from preprocessing.preprocess import (
    preprocess_text,
    tokenize,
    remove_stopwords,
    lemmatize_tokens,
    preprocess_pipeline,
    tfidf_vectorizer,
)


class TestPreprocessText:
    def test_lowercases_text(self):
        assert preprocess_text("HELLO WORLD") == "hello world"

    def test_removes_urls(self):
        result = preprocess_text("visit https://example.com for more")
        assert "https" not in result
        assert "example" not in result

    def test_removes_www_urls(self):
        result = preprocess_text("go to www.example.com today")
        assert "www" not in result

    def test_removes_emails(self):
        result = preprocess_text("contact us at test@example.com")
        assert "@" not in result

    def test_removes_punctuation(self):
        result = preprocess_text("hello, world! how are you?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_removes_digits(self):
        result = preprocess_text("there are 42 items in 2024")
        assert "42" not in result
        assert "2024" not in result

    def test_collapses_whitespace(self):
        result = preprocess_text("  too   many   spaces  ")
        assert "  " not in result
        assert result == result.strip()

    def test_empty_string(self):
        assert preprocess_text("") == ""


class TestTokenize:
    def test_splits_words(self):
        tokens = tokenize("hello world")
        assert "hello" in tokens
        assert "world" in tokens

    def test_returns_list(self):
        assert isinstance(tokenize("some text"), list)


class TestRemoveStopwords:
    def test_removes_common_stopwords(self):
        tokens = ["the", "quick", "brown", "fox", "is", "a"]
        result = remove_stopwords(tokens)
        assert "the" not in result
        assert "is" not in result
        assert "a" not in result

    def test_keeps_meaningful_words(self):
        tokens = ["the", "president", "announced", "policy"]
        result = remove_stopwords(tokens)
        assert "president" in result
        assert "announced" in result
        assert "policy" in result

    def test_empty_list(self):
        assert remove_stopwords([]) == []


class TestLemmatizeTokens:
    def test_lemmatizes_plural(self):
        result = lemmatize_tokens(["dogs", "running", "better"])
        assert "dog" in result

    def test_returns_list(self):
        assert isinstance(lemmatize_tokens(["words"]), list)

    def test_empty_list(self):
        assert lemmatize_tokens([]) == []


class TestPreprocessPipeline:
    def test_returns_string(self):
        result = preprocess_pipeline("The president announced new policies today!")
        assert isinstance(result, str)

    def test_no_stopwords_in_output(self):
        result = preprocess_pipeline("This is a test of the pipeline")
        tokens = result.split()
        stopwords_sample = {"this", "is", "a", "of", "the"}
        assert stopwords_sample.isdisjoint(set(tokens))

    def test_no_urls_in_output(self):
        result = preprocess_pipeline("Read more at https://news.com/article")
        assert "https" not in result

    def test_empty_string(self):
        result = preprocess_pipeline("")
        assert isinstance(result, str)


# Shared fixture: identical docs guarantee every unigram/bigram appears in all 6
# documents, satisfying tfidf_vectorizer's min_df=5 constraint.
_BASE = "president announced economy policy trade deal signed"
_TFIDF_DOCS = [_BASE] * 6


class TestTfidfVectorizer:
    def test_matrix_shape_matches_vocab(self):
        df = pd.DataFrame({"processed_content": _TFIDF_DOCS})
        matrix, vectorizer = tfidf_vectorizer(df)
        assert matrix.shape[0] == len(_TFIDF_DOCS)
        assert matrix.shape[1] <= 5000

    def test_vectorizer_has_feature_names(self):
        df = pd.DataFrame({"processed_content": _TFIDF_DOCS})
        _, vectorizer = tfidf_vectorizer(df)
        assert len(vectorizer.get_feature_names_out()) > 0

    def test_sparse_matrix_returned(self):
        import scipy.sparse as sp
        df = pd.DataFrame({"processed_content": _TFIDF_DOCS})
        matrix, _ = tfidf_vectorizer(df)
        assert sp.issparse(matrix)
