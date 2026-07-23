"""Fit final models on the full dataset and persist artifacts for serving.

Unlike train_classical.py (cross-validation for metric estimates) and
evaluate.py (train/test split for error analysis), this script trains on
ALL available data since the goal here is the best possible model to ship,
not an unbiased performance estimate (that estimate already exists from the
earlier steps).
"""

from pathlib import Path

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import build_structural_features, clean_text, ensure_nltk_data

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "emails.parquet"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RANDOM_STATE = 42


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PATH)
    y = (df["label"] == "spam").astype(int).to_numpy()

    ensure_nltk_data()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    print("Cleaning text + fitting TF-IDF vectorizer...")
    cleaned = df["body"].apply(lambda t: clean_text(t, stop_words, lemmatizer))
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(cleaned)
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")

    print("Building structural features + fitting scaler...")
    structural = build_structural_features(df)
    scaler = StandardScaler()
    structural_scaled = scaler.fit_transform(structural)
    joblib.dump(scaler, MODELS_DIR / "structural_scaler.joblib")
    joblib.dump(list(structural.columns), MODELS_DIR / "structural_columns.joblib")

    combined = sparse.hstack([tfidf, sparse.csr_matrix(structural_scaled)]).tocsr()

    print("Training Multinomial NB (TF-IDF)...")
    nb = MultinomialNB()
    nb.fit(tfidf, y)
    joblib.dump(nb, MODELS_DIR / "nb_tfidf.joblib")

    print("Training Logistic Regression (TF-IDF)...")
    logreg_tfidf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    logreg_tfidf.fit(tfidf, y)
    joblib.dump(logreg_tfidf, MODELS_DIR / "logreg_tfidf.joblib")

    print("Training Logistic Regression (combined)...")
    logreg_combined = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    logreg_combined.fit(combined, y)
    joblib.dump(logreg_combined, MODELS_DIR / "logreg_combined.joblib")

    print("Training XGBoost (combined)...")
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, eval_metric="logloss", random_state=RANDOM_STATE)
    xgb.fit(combined, y)
    joblib.dump(xgb, MODELS_DIR / "xgboost_combined.joblib")

    print(f"\nAll artifacts saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
