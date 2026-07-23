"""Evaluation report: confusion matrices, ROC/PR curves, and error analysis.

Unlike train_classical.py (which cross-validates for a robust metric estimate),
this script uses a single stratified train/test split so we have concrete
held-out predictions to inspect — needed for confusion matrices, curves, and
pulling actual misclassified emails for error analysis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data():
    emails_df = pd.read_parquet(DATA_DIR / "emails.parquet")
    labels_df = pd.read_parquet(DATA_DIR / "labels.parquet")
    y = (labels_df["label"] == "spam").astype(int).to_numpy()

    tfidf = sparse.load_npz(DATA_DIR / "features_tfidf.npz")
    structural = pd.read_parquet(DATA_DIR / "features_structural.parquet")

    return emails_df, tfidf, structural, y


def make_combined(tfidf, structural, train_idx, test_idx):
    scaler = StandardScaler()
    struct_train_scaled = scaler.fit_transform(structural.iloc[train_idx])
    struct_test_scaled = scaler.transform(structural.iloc[test_idx])

    X_train = sparse.hstack([tfidf[train_idx], sparse.csr_matrix(struct_train_scaled)]).tocsr()
    X_test = sparse.hstack([tfidf[test_idx], sparse.csr_matrix(struct_test_scaled)]).tocsr()
    return X_train, X_test


def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    out_path = FIGURES_DIR / f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_roc_pr(y_test, y_proba, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[0], name=model_name)
    axes[0].set_title(f"ROC Curve — {model_name}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)

    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=axes[1], name=model_name)
    axes[1].set_title(f"Precision-Recall Curve — {model_name}")

    fig.tight_layout()
    out_path = FIGURES_DIR / f"roc_pr_{model_name.replace(' ', '_').lower()}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def collect_misclassified(emails_df, test_idx, y_test, y_pred, model_name, n=8):
    """Pull a sample of false positives (ham called spam) and false negatives
    (spam called ham) with enough context to spot patterns."""
    test_emails = emails_df.iloc[test_idx].reset_index(drop=True)
    results = pd.DataFrame({
        "true_label": ["spam" if v == 1 else "ham" for v in y_test],
        "pred_label": ["spam" if v == 1 else "ham" for v in y_pred],
    })
    combined = pd.concat([test_emails[["source_folder", "from", "subject", "body"]].reset_index(drop=True), results], axis=1)

    false_positives = combined[(combined["true_label"] == "ham") & (combined["pred_label"] == "spam")]
    false_negatives = combined[(combined["true_label"] == "spam") & (combined["pred_label"] == "ham")]

    fp_sample = false_positives.head(n).copy()
    fp_sample["error_type"] = "false_positive (ham -> spam)"
    fn_sample = false_negatives.head(n).copy()
    fn_sample["error_type"] = "false_negative (spam -> ham)"

    sample = pd.concat([fp_sample, fn_sample], ignore_index=True)
    sample["model"] = model_name
    sample["body"] = sample["body"].str.slice(0, 200)

    return sample, len(false_positives), len(false_negatives)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    emails_df, tfidf, structural, y = load_data()
    indices = range(len(y))
    train_idx, test_idx, y_train, y_test = train_test_split(
        list(indices), y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    all_misclassified = []

    # --- Model 1: Multinomial NB on TF-IDF ---
    X_train, X_test = tfidf[train_idx], tfidf[test_idx]
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    y_proba = nb.predict_proba(X_test)[:, 1]

    plot_confusion_matrix(y_test, y_pred, "Naive Bayes (TF-IDF)")
    plot_roc_pr(y_test, y_proba, "Naive Bayes (TF-IDF)")
    sample, n_fp, n_fn = collect_misclassified(emails_df, test_idx, y_test, y_pred, "Naive Bayes (TF-IDF)")
    print(f"Naive Bayes (TF-IDF): {n_fp} false positives, {n_fn} false negatives")
    all_misclassified.append(sample)

    # --- Model 2: Logistic Regression on TF-IDF ---
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_proba = logreg.predict_proba(X_test)[:, 1]

    plot_confusion_matrix(y_test, y_pred, "Logistic Regression (TF-IDF)")
    plot_roc_pr(y_test, y_proba, "Logistic Regression (TF-IDF)")
    sample, n_fp, n_fn = collect_misclassified(emails_df, test_idx, y_test, y_pred, "Logistic Regression (TF-IDF)")
    print(f"Logistic Regression (TF-IDF): {n_fp} false positives, {n_fn} false negatives")
    all_misclassified.append(sample)

    # --- Model 3 & 4: combined features ---
    X_train_comb, X_test_comb = make_combined(tfidf, structural, train_idx, test_idx)

    logreg_comb = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    logreg_comb.fit(X_train_comb, y_train)
    y_pred = logreg_comb.predict(X_test_comb)
    y_proba = logreg_comb.predict_proba(X_test_comb)[:, 1]

    plot_confusion_matrix(y_test, y_pred, "Logistic Regression (Combined)")
    plot_roc_pr(y_test, y_proba, "Logistic Regression (Combined)")
    sample, n_fp, n_fn = collect_misclassified(emails_df, test_idx, y_test, y_pred, "Logistic Regression (Combined)")
    print(f"Logistic Regression (Combined): {n_fp} false positives, {n_fn} false negatives")
    all_misclassified.append(sample)

    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, eval_metric="logloss", random_state=RANDOM_STATE)
    xgb.fit(X_train_comb, y_train)
    y_pred = xgb.predict(X_test_comb)
    y_proba = xgb.predict_proba(X_test_comb)[:, 1]

    plot_confusion_matrix(y_test, y_pred, "XGBoost (Combined)")
    plot_roc_pr(y_test, y_proba, "XGBoost (Combined)")
    sample, n_fp, n_fn = collect_misclassified(emails_df, test_idx, y_test, y_pred, "XGBoost (Combined)")
    print(f"XGBoost (Combined): {n_fp} false positives, {n_fn} false negatives")
    all_misclassified.append(sample)

    error_report = pd.concat(all_misclassified, ignore_index=True)
    out_path = REPORTS_DIR / "misclassified_examples.csv"
    error_report.to_csv(out_path, index=False)
    print(f"\nMisclassified examples saved -> {out_path}")
    print(f"Figures saved -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
