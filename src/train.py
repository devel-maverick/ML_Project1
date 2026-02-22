import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.preprocess import load_data, preprocess_entire_dataframe
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from config import DATA_PATH, MODEL_DIR
import joblib

df = load_data(DATA_PATH)
print("Dataset shape:", df.shape)
print("Label distribution:\n", df["label"].value_counts())

df = preprocess_entire_dataframe(df)
X_text = df["processed_content"]
y = df["label"]
X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size = 0.2, random_state = 42)

vectorizer = TfidfVectorizer(max_features = 5000, ngram_range = (1, 2), min_df = 5)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


model = LogisticRegression(max_iter = 1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

MODEL_DIR.mkdir(exist_ok = True)

joblib.dump(model, MODEL_DIR / "classifier.pkl")
joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")
print(f"\nModel saved to {MODEL_DIR / 'classifier.pkl'}")
print(f"Vectorizer saved to {MODEL_DIR / 'vectorizer.pkl'}")
