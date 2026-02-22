import joblib
import numpy as np

# Load trained model and vectorizer
model = joblib.load("models/classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

print("📰 Fake News Detection System")
print("----------------------------------")

while True:
    text = input("\nEnter news text (type 'exit' to stop): ")

    if text.lower() == "exit":
        break

    # Convert text to TF-IDF
    tfidf_vector = vectorizer.transform([text])

    # Prediction
    prediction = model.predict(tfidf_vector)[0]
    probability = model.predict_proba(tfidf_vector)[0]
    confidence = max(probability) * 100

    # Print result
    if prediction == 1:
        print(f"\nPrediction: REAL NEWS ✅ ({confidence:.2f}% confidence)")
    else:
        print(f"\nPrediction: FAKE NEWS ❌ ({confidence:.2f}% confidence)")

    # ===== Explanation Part =====
    print("\nTop Influencing Words:")

    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    indices = tfidf_vector.nonzero()[1]

    contributions = []

    for idx in indices:
        word = feature_names[idx]
        weight = coefficients[idx]
        value = tfidf_vector[0, idx]
        contribution = weight * value
        contributions.append((word, contribution))

    # Sort by strongest influence
    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)

    for word, score in contributions[:10]:
        direction = "REAL ↑" if score > 0 else "FAKE ↓"
        print(f"{word} → {direction} (score: {score:.4f})")