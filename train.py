"""
train.py
--------
Train the Spam Email Classifier.
Run once before launching the app: python train.py

Dataset: SMS Spam Collection (UCI Machine Learning Repository)
Auto-downloads via requests if not present locally.
"""

import os
import re
import pickle
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("data")
MODEL_DIR = Path("model")
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DATASET_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATASET_ZIP  = DATA_DIR / "smsspamcollection.zip"
DATASET_FILE = DATA_DIR / "SMSSpamCollection"

MODEL_PATH      = MODEL_DIR / "spam_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"


# ── Download Dataset ──────────────────────────────────────────────────────────
def download_dataset():
    if DATASET_FILE.exists():
        print("✅ Dataset already exists.")
        return

    print("📥 Downloading SMS Spam Collection dataset...")
    urllib.request.urlretrieve(DATASET_URL, DATASET_ZIP)

    with zipfile.ZipFile(DATASET_ZIP, "r") as z:
        z.extractall(DATA_DIR)

    print(f"✅ Dataset saved to {DATASET_FILE}")


# ── Load Data ─────────────────────────────────────────────────────────────────
def load_data():
    texts, labels = [], []
    with open(DATASET_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                label, text = parts
                labels.append(label)
                texts.append(text)
    return texts, labels


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    print("\n" + "="*55)
    print("  📧 Spam Classifier — Training")
    print("="*55)

    download_dataset()

    print("\n📦 Loading data...")
    texts, labels = load_data()
    print(f"  Total samples : {len(texts)}")
    print(f"  Spam          : {labels.count('spam')}")
    print(f"  Ham (not spam): {labels.count('ham')}")

    # Preprocess
    cleaned = [preprocess(t) for t in texts]

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # ── TF-IDF Vectorizer ─────────────────────────────────────
    print("\n🔢 Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),    # unigrams + bigrams
        sublinear_tf=True,     # apply log scaling
        min_df=2               # ignore very rare terms
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")

    # ── Naive Bayes ───────────────────────────────────────────
    print("\n🤖 Training Multinomial Naive Bayes...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vec, y_train)

    # ── Evaluation ────────────────────────────────────────────
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n📊 Test Set Results:")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"\n{classification_report(y_test, y_pred, target_names=['ham', 'spam'])}")

    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    print(f"            Predicted Ham  Predicted Spam")
    print(f"  Actual Ham    {cm[0][0]:>5}          {cm[0][1]:>5}")
    print(f"  Actual Spam   {cm[1][0]:>5}          {cm[1][1]:>5}")

    # Cross-validation
    print("\n🔄 5-Fold Cross-Validation (on full dataset)...")
    all_vec = vectorizer.transform(cleaned)
    cv_scores = cross_val_score(model, all_vec, labels, cv=5, scoring="accuracy")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Save ──────────────────────────────────────────────────
    print("\n💾 Saving model and vectorizer...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"  ✅ Model saved     → {MODEL_PATH}")
    print(f"  ✅ Vectorizer saved → {VECTORIZER_PATH}")

    print("\n" + "="*55)
    print("  Training complete! Run: streamlit run app.py")
    print("="*55 + "\n")


if __name__ == "__main__":
    train()
