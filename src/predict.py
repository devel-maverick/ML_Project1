"""predict.py — Load the trained classifier and expose a prediction interface."""

import joblib
from preprocessing.preprocess import preprocess_pipeline
from config import MODEL_DIR
model = joblib.load(MODEL_DIR / "classifier.pkl")
vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")

def predict_article(text):
    """Classify a raw news article string.

    Parameters
    ----------
    text : str
        Raw article text (title + body).

    Returns
    -------
    dict
        ``{"prediction": int, "confidence": float, "probabilities": list[float]}``
        where *prediction* is 1 (real) or 0 (fake).
    """
    processed = preprocess_pipeline(text)
    vector = vectorizer.transform([processed])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]
    confidence = max(probability)
    return {
        "prediction": int(prediction),
        "confidence": float(confidence),
        "probabilities": probability.tolist()
    }

def get_model_and_vectorizer():
    """Return the loaded (model, vectorizer) tuple for external use."""
    return model, vectorizer