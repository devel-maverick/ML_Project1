"""Unit tests for src/predict.py and src/explain.py."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestPredictArticle:
    """Tests for predict_article using a mocked model and vectorizer."""

    def _make_mocks(self, pred_class=1, proba=None):
        """Return a (model_mock, vectorizer_mock) pair."""
        if proba is None:
            proba = [0.1, 0.9]

        vectorizer = MagicMock()
        vectorizer.transform.return_value = MagicMock()

        model = MagicMock()
        model.predict.return_value = np.array([pred_class])
        model.predict_proba.return_value = np.array([proba])

        return model, vectorizer

    def test_returns_dict_with_required_keys(self):
        model, vectorizer = self._make_mocks()
        with patch("src.predict.model", model), patch("src.predict.vectorizer", vectorizer):
            from src.predict import predict_article
            result = predict_article("Some news text here")
        assert "prediction" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_prediction_is_int(self):
        model, vectorizer = self._make_mocks(pred_class=1)
        with patch("src.predict.model", model), patch("src.predict.vectorizer", vectorizer):
            from src.predict import predict_article
            result = predict_article("Real news article")
        assert isinstance(result["prediction"], int)

    def test_confidence_between_0_and_1(self):
        model, vectorizer = self._make_mocks(proba=[0.2, 0.8])
        with patch("src.predict.model", model), patch("src.predict.vectorizer", vectorizer):
            from src.predict import predict_article
            result = predict_article("Some text")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_to_1(self):
        model, vectorizer = self._make_mocks(proba=[0.35, 0.65])
        with patch("src.predict.model", model), patch("src.predict.vectorizer", vectorizer):
            from src.predict import predict_article
            result = predict_article("Some text")
        assert abs(sum(result["probabilities"]) - 1.0) < 1e-6

    def test_fake_prediction(self):
        model, vectorizer = self._make_mocks(pred_class=0, proba=[0.8, 0.2])
        with patch("src.predict.model", model), patch("src.predict.vectorizer", vectorizer):
            from src.predict import predict_article
            result = predict_article("Sensational fake clickbait headline")
        assert result["prediction"] == 0
        assert result["confidence"] == pytest.approx(0.8)

    def test_real_prediction(self):
        model, vectorizer = self._make_mocks(pred_class=1, proba=[0.05, 0.95])
        with patch("src.predict.model", model), patch("src.predict.vectorizer", vectorizer):
            from src.predict import predict_article
            result = predict_article("Verified news from Reuters")
        assert result["prediction"] == 1
        assert result["confidence"] == pytest.approx(0.95)


class TestExplainPrediction:
    """Tests for explain_prediction in src/explain.py."""

    def _make_explain_mocks(self, feature_names, coef, tfidf_values):
        """Build mock model and vectorizer for explain_prediction."""
        import scipy.sparse as sp

        vectorizer = MagicMock()
        # Build a sparse vector with given column indices having given values
        n_features = len(feature_names)
        nonzero_cols = list(range(len(tfidf_values)))
        data = np.array(tfidf_values, dtype=float)
        row = np.zeros(len(nonzero_cols), dtype=int)
        col = np.array(nonzero_cols, dtype=int)
        sparse_vec = sp.csr_matrix((data, (row, col)), shape=(1, n_features))
        vectorizer.transform.return_value = sparse_vec
        vectorizer.get_feature_names_out.return_value = np.array(feature_names)

        model = MagicMock()
        model.coef_ = [np.array(coef)]

        return model, vectorizer

    def test_returns_list(self):
        from src.explain import explain_prediction
        feat = ["president", "fake", "economy", "war"]
        coef = [0.5, -0.8, 0.3, -0.2]
        vals = [0.4, 0.6, 0.1, 0.3]
        model, vectorizer = self._make_explain_mocks(feat, coef, vals)
        result = explain_prediction("some text", model, vectorizer, top_n=3)
        assert isinstance(result, list)

    def test_respects_top_n(self):
        from src.explain import explain_prediction
        feat = ["a", "b", "c", "d", "e"]
        coef = [0.9, -0.7, 0.5, -0.3, 0.1]
        vals = [0.5, 0.5, 0.5, 0.5, 0.5]
        model, vectorizer = self._make_explain_mocks(feat, coef, vals)
        result = explain_prediction("text", model, vectorizer, top_n=3)
        assert len(result) <= 3

    def test_sorted_by_absolute_contribution(self):
        from src.explain import explain_prediction
        feat = ["small", "large", "medium"]
        coef = [0.1, -0.9, 0.4]
        vals = [1.0, 1.0, 1.0]
        model, vectorizer = self._make_explain_mocks(feat, coef, vals)
        result = explain_prediction("text", model, vectorizer, top_n=3)
        scores = [abs(s) for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_returns_tuples_of_word_and_score(self):
        from src.explain import explain_prediction
        feat = ["word"]
        coef = [0.5]
        vals = [0.8]
        model, vectorizer = self._make_explain_mocks(feat, coef, vals)
        result = explain_prediction("text", model, vectorizer, top_n=1)
        assert len(result) == 1
        word, score = result[0]
        assert isinstance(word, str)
        assert isinstance(score, float)
