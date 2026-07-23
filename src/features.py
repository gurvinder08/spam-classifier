"""Feature engineering for the spam classifier.

Builds two independent feature representations from the parsed email
dataset (data/processed/emails.parquet) so they can be evaluated alone
or combined later:

  - Text-based:   TF-IDF over cleaned email bodies
  - Structural:   handcrafted signals (caps ratio, urgency words, links, ...)
"""

import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "emails.parquet"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")

URGENCY_KEYWORDS = [
    "urgent", "act now", "limited time", "click here", "call now",
    "winner", "congratulations", "free", "guarantee", "cash",
    "risk free", "act immediately", "expires", "offer expires",
    "don't delete", "you have been selected", "verify your account",
]


def ensure_nltk_data():
    for resource in ["corpora/stopwords", "corpora/wordnet", "corpora/omw-1.4"]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1], quiet=True)


# ---------------------------------------------------------------------------
# Text cleaning + TF-IDF
# ---------------------------------------------------------------------------

def clean_text(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> str:
    text = text or ""
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = text.lower()
    text = NON_ALPHA_RE.sub(" ", text)

    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)


def build_tfidf_features(df: pd.DataFrame, max_features: int = 5000):
    ensure_nltk_data()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    cleaned = df["body"].apply(lambda t: clean_text(t, stop_words, lemmatizer))

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(cleaned)

    return tfidf_matrix, vectorizer


# ---------------------------------------------------------------------------
# Structural / handcrafted features
# ---------------------------------------------------------------------------

def extract_domain(addr: str) -> str:
    if not isinstance(addr, str):
        return ""
    match = re.search(r"@([\w.-]+)", addr)
    return match.group(1).lower() if match else ""


def capital_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def has_urgency_language(text: str) -> int:
    text_lower = (text or "").lower()
    return int(any(kw in text_lower for kw in URGENCY_KEYWORDS))


def build_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    body = df["body"].fillna("")
    subject = df["subject"].fillna("")

    from_domain = df["from"].apply(extract_domain)
    reply_to_domain = df["reply_to"].apply(extract_domain)

    features = pd.DataFrame(index=df.index)
    features["capital_ratio"] = body.apply(capital_ratio)
    features["subject_capital_ratio"] = subject.apply(capital_ratio)
    features["exclamation_count"] = body.str.count("!")
    features["subject_exclamation_count"] = subject.str.count("!")
    features["subject_length"] = subject.str.len()
    features["body_length"] = body.str.len()
    features["has_url"] = body.str.contains(URL_RE).astype(int)
    features["link_count"] = body.apply(lambda t: len(URL_RE.findall(t)))
    features["urgency_language"] = body.apply(has_urgency_language)
    features["sender_replyto_mismatch"] = (
        (from_domain != "") & (reply_to_domain != "") & (from_domain != reply_to_domain)
    ).astype(int)
    features["html_only_body"] = body.str.contains(HTML_TAG_RE).astype(int)
    features["received_count"] = df["received_count"]

    return features


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    df = pd.read_parquet(DATA_PATH)

    print("Building structural features...")
    structural_df = build_structural_features(df)
    structural_path = OUT_DIR / "features_structural.parquet"
    structural_df.to_parquet(structural_path, index=False)
    print(f"Saved structural features {structural_df.shape} -> {structural_path}")

    print("\nBuilding TF-IDF features...")
    tfidf_matrix, vectorizer = build_tfidf_features(df)
    tfidf_path = OUT_DIR / "features_tfidf.npz"
    sparse.save_npz(tfidf_path, tfidf_matrix)
    print(f"Saved TF-IDF matrix {tfidf_matrix.shape} -> {tfidf_path}")

    vocab_path = OUT_DIR / "tfidf_vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("\n".join(vectorizer.get_feature_names_out()))
    print(f"Saved vocabulary ({len(vectorizer.vocabulary_)} terms) -> {vocab_path}")

    labels_path = OUT_DIR / "labels.parquet"
    df[["label"]].to_parquet(labels_path, index=False)
    print(f"Saved labels -> {labels_path}")


if __name__ == "__main__":
    main()
