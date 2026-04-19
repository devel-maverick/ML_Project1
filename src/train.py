"""
train.py — Train a Logistic Regression fake-news classifier.

Steps
-----
1. Load and preprocess the WELFake dataset.
2. Vectorise text with TF-IDF (unigrams + bigrams, max 5 000 features).
3. Perform 5-fold cross-validation and report mean CV accuracy.
4. Train a final model on the full training split and evaluate on the test split.
5. Save confusion-matrix and ROC-curve plots to reports/.
6. Persist the trained model and vectorizer to models/.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from sklearn.model_selection import cross_val_score, train_test_split

from config import DATA_PATH, MODEL_DIR
from preprocessing.preprocess import load_data, preprocess_entire_dataframe

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def save_confusion_matrix(y_true, y_pred, output_path):
    """Render and save a confusion-matrix heatmap to *output_path*."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {output_path}")


def save_roc_curve(model, X_test, y_test, output_path):
    """Compute and save an ROC curve plot to *output_path*."""
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"ROC curve saved        → {output_path}")


def main():
    """Full training pipeline: load → preprocess → vectorise → train → evaluate → save."""
    df = load_data(DATA_PATH)
    print("Dataset shape:", df.shape)
    print("Label distribution:\n", df["label"].value_counts())

    df = preprocess_entire_dataframe(df)
    X_text = df["processed_content"]
    y = df["label"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = LogisticRegression(max_iter=1000)

    print("\nRunning 5-fold cross-validation…")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Per-fold:    {np.round(cv_scores, 4)}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nTrain Accuracy:", model.score(X_train, y_train))
    print("Test  Accuracy:", model.score(X_test, y_test))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    REPORTS_DIR.mkdir(exist_ok=True)
    save_confusion_matrix(y_test, y_pred, REPORTS_DIR / "confusion_matrix.png")
    save_roc_curve(model, X_test, y_test, REPORTS_DIR / "roc_curve.png")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "classifier.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")
    print(f"\nModel saved     → {MODEL_DIR / 'classifier.pkl'}")
    print(f"Vectorizer saved → {MODEL_DIR / 'vectorizer.pkl'}")


if __name__ == "__main__":
    main()
