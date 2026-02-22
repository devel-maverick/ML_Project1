import numpy as np

def explain_prediction(text, model, vectorizer, top_n=10):

    # Transform text
    tfidf_vector = vectorizer.transform([text])

    # Get feature names and weights
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    # Get non-zero features from this text
    indices = tfidf_vector.nonzero()[1]

    # Calculate contribution for each word
    contributions = []

    for idx in indices:
        word = feature_names[idx]
        weight = coefficients[idx]
        value = tfidf_vector[0, idx]
        contribution = weight * value
        contributions.append((word, contribution))

    # Sort by strongest influence
    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)

    return contributions[:top_n]