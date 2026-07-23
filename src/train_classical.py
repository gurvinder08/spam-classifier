"""Train and cross-validate baseline classical models for spam classification.

Models (in report order):
  1. Multinomial Naive Bayes      on TF-IDF only
  2. Logistic Regression          on TF-IDF only
  3. Logistic Regression          on TF-IDF + handcrafted (combined)
  4. XGBoost                      on TF-IDF + handcrafted (combined)

Uses stratified k-fold CV. Metrics reported per fold-average, priority order:
  spam precision, ham false-positive rate, F1 (spam), accuracy (last, least trusted).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
N_SPLITS = 5
RANDOM_STATE = 42


def load_data():
    labels_df = pd.read_parquet(DATA_DIR / "labels.parquet")
    y = (labels_df["label"] == "spam").astype(int).to_numpy()

    tfidf = sparse.load_npz(DATA_DIR / "features_tfidf.npz")
    structural = pd.read_parquet(DATA_DIR / "features_structural.parquet")

    return tfidf, structural, y


def false_positive_rate(y_true, y_pred):
    """Fraction of ham (0) incorrectly predicted as spam (1)."""
    ham_mask = y_true == 0
    if ham_mask.sum() == 0:
        return 0.0
    return (y_pred[ham_mask] == 1).mean()


def evaluate_model(model_name, X, y, skf, scale_dense_cols=None):
    from sklearn.metrics import accuracy_score, f1_score, precision_score

    fold_metrics = {"precision_spam": [], "fpr_ham": [], "f1_spam": [], "accuracy": []}

    for train_idx, test_idx in skf.split(np.arange(len(y)), y):
        if isinstance(X, tuple):
            # (tfidf_sparse, structural_dense) combined case
            tfidf_x, struct_x = X
            X_train_tfidf, X_test_tfidf = tfidf_x[train_idx], tfidf_x[test_idx]
            X_train_struct, X_test_struct = struct_x.iloc[train_idx], struct_x.iloc[test_idx]

            scaler = StandardScaler()
            X_train_struct_scaled = scaler.fit_transform(X_train_struct)
            X_test_struct_scaled = scaler.transform(X_test_struct)

            X_train = sparse.hstack([X_train_tfidf, sparse.csr_matrix(X_train_struct_scaled)]).tocsr()
            X_test = sparse.hstack([X_test_tfidf, sparse.csr_matrix(X_test_struct_scaled)]).tocsr()
        else:
            X_train, X_test = X[train_idx], X[test_idx]

        y_train, y_test = y[train_idx], y[test_idx]

        if model_name == "nb":
            clf = MultinomialNB()
        elif model_name == "logreg":
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        elif model_name == "xgb":
            clf = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                eval_metric="logloss", random_state=RANDOM_STATE,
            )
        else:
            raise ValueError(model_name)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        fold_metrics["precision_spam"].append(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        fold_metrics["fpr_ham"].append(false_positive_rate(y_test, y_pred))
        fold_metrics["f1_spam"].append(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
        fold_metrics["accuracy"].append(accuracy_score(y_test, y_pred))

    return {k: (np.mean(v), np.std(v)) for k, v in fold_metrics.items()}


def print_results(name, metrics):
    print(f"\n{name}")
    print(f"  Spam precision : {metrics['precision_spam'][0]:.3f} (+/- {metrics['precision_spam'][1]:.3f})")
    print(f"  Ham FPR        : {metrics['fpr_ham'][0]:.3f} (+/- {metrics['fpr_ham'][1]:.3f})")
    print(f"  Spam F1        : {metrics['f1_spam'][0]:.3f} (+/- {metrics['f1_spam'][1]:.3f})")
    print(f"  Accuracy       : {metrics['accuracy'][0]:.3f} (+/- {metrics['accuracy'][1]:.3f})")


def main():
    tfidf, structural, y = load_data()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results_rows = []

    nb_metrics = evaluate_model("nb", tfidf, y, skf)
    print_results("1. Multinomial Naive Bayes (TF-IDF)", nb_metrics)
    results_rows.append(("Multinomial NB", "TF-IDF", nb_metrics))

    logreg_tfidf_metrics = evaluate_model("logreg", tfidf, y, skf)
    print_results("2. Logistic Regression (TF-IDF)", logreg_tfidf_metrics)
    results_rows.append(("Logistic Regression", "TF-IDF", logreg_tfidf_metrics))

    logreg_combined_metrics = evaluate_model("logreg", (tfidf, structural), y, skf)
    print_results("3. Logistic Regression (TF-IDF + handcrafted)", logreg_combined_metrics)
    results_rows.append(("Logistic Regression", "TF-IDF + handcrafted", logreg_combined_metrics))

    xgb_combined_metrics = evaluate_model("xgb", (tfidf, structural), y, skf)
    print_results("4. XGBoost (TF-IDF + handcrafted)", xgb_combined_metrics)
    results_rows.append(("XGBoost", "TF-IDF + handcrafted", xgb_combined_metrics))

    summary = pd.DataFrame([
        {
            "model": model,
            "features": feats,
            "spam_precision": m["precision_spam"][0],
            "ham_fpr": m["fpr_ham"][0],
            "spam_f1": m["f1_spam"][0],
            "accuracy": m["accuracy"][0],
        }
        for model, feats, m in results_rows
    ])

    out_path = DATA_DIR / "baseline_results.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nResults table saved -> {out_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
