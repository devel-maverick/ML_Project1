import joblib
from preprocessing.preprocess import preprocess_pipeline
from config import MODEL_DIR

model = joblib.load(MODEL_DIR / "classifier.pkl")
vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")

def predict_article(text):
    processed = preprocess_pipeline(text)
    vector = vectorizer.transform([processed])

    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]

    confidence = max(probability)

    return {
    "prediction": prediction,
    "confidence": confidence,
    "probabilities": probability
}