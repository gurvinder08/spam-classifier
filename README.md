# Email Spam Classifier

A spam/ham classifier built on the public SpamAssassin corpus, comparing classic
text-based and structural feature approaches, with a FastAPI service for inference.

## Problem

Classify raw emails as **spam** or **ham** (legitimate). The dataset is imbalanced
(~69% ham / 31% spam) and includes a deliberately hard category (`hard_ham`) —
legitimate bulk/newsletter mail that structurally resembles spam. Because of the
imbalance, **accuracy is a misleading metric**: a model that always predicts "ham"
would score ~69% accuracy while catching zero spam. The metrics that actually
matter here are:

1. **Spam precision** — of the emails flagged as spam, how many really are spam
   (low precision = legitimate mail wrongly blocked, the costliest error type)
2. **Ham false-positive rate** — of all legitimate mail, how much gets
   misclassified as spam
3. F1 and accuracy, reported but treated as secondary

## Dataset

[SpamAssassin public corpus](https://spamassassin.apache.org/old/publiccorpus/) —
6,100 raw `.eml` files across 5 folders:

| Folder | Label | Count |
|---|---|---|
| easy_ham | ham | 2,551 |
| easy_ham_2 | ham | 1,401 |
| hard_ham | ham | 250 |
| spam | spam | 501 |
| spam_2 | spam | 1,397 |
| **Total** | | **6,100** (4,202 ham / 1,898 spam) |

Raw files are not committed to this repo (see [How to Run](#how-to-run) to
re-download them) — only code and derived artifacts are tracked.

## Approach

1. **Parsing** (`src/parse_emails.py`) — walk all 5 folders, parse each `.eml`
   with Python's `email` module, extract headers (From, To, Subject, Date,
   Reply-To, Received count) and body (text/plain, falling back to stripped
   text/html). Handles malformed charset headers found in ~0.4% of the spam
   files (`unknown-8bit`, `default`, `default_charset`) by falling back to a
   UTF-8 decode instead of dropping the email. Output: `data/processed/emails.parquet`.

2. **EDA** (`notebooks/eda.ipynb`) — class balance, body length distribution,
   HTML-vs-plain-text ratio, subject/sender patterns, missing-header audit.

3. **Feature engineering** (`src/features.py`) — two independent feature sets,
   kept separate so they can be evaluated alone or combined:
   - **TF-IDF**: cleaned body text (HTML stripped, URLs stripped, lowercased,
     stopwords removed, lemmatized) → unigrams + bigrams, top 5,000 terms
   - **Structural**: capital-letter ratio, exclamation counts, subject/body
     length, URL count, urgency-keyword flag, sender/reply-to domain mismatch,
     HTML-only body flag, received-header count

4. **Baseline models** (`src/train_classical.py`) — stratified 5-fold
   cross-validation across 4 configurations (see results below).

5. **Evaluation** (`src/evaluate.py`) — held-out 80/20 split for confusion
   matrices, ROC/PR curves, and pulling actual misclassified emails for error
   analysis (`reports/`).

6. **Serving** (`src/api/main.py`) — FastAPI service loading all trained
   models, selectable per request.

*(LLM zero-shot comparison was scoped but skipped — no reliable free-tier LLM
API was available at the time.)*

## Tools & Techniques

A quick reference for the less obvious pieces of the pipeline:

- **TF-IDF (Term Frequency–Inverse Document Frequency)** — turns each email
  body into a vector of word/phrase weights. A word gets a high score in a
  document if it appears often *there* but rarely across the rest of the
  corpus — so it highlights words that are actually distinctive to that
  email, rather than common words that appear everywhere (which plain word
  counts would over-weight).
- **Lemmatization** — reduces words to their dictionary base form (e.g.
  "winning", "won", "wins" → "win") so the model treats them as the same
  feature instead of splitting the signal across variants. Done with NLTK's
  `WordNetLemmatizer`.
- **Stopword removal** — strips very common words ("the", "is", "and", ...)
  that carry no discriminative signal between spam and ham, before TF-IDF is
  computed.
- **N-grams (unigrams + bigrams)** — the TF-IDF vocabulary includes both
  single words and two-word phrases (e.g. "click here", "act now"), since
  some spam signals are phrase-level, not single-word.
- **Structural/handcrafted features** — signals engineered from domain
  knowledge about what spam tends to look like, independent of specific
  words: ratio of capital letters, exclamation mark counts, presence of
  urgency phrases, number of links, whether the sender and reply-to domains
  mismatch, etc. These complement TF-IDF because they capture *style and
  structure* rather than vocabulary.
- **Stratified k-fold cross-validation** — splits the data into k folds for
  training/testing while preserving the overall class ratio in every fold.
  Necessary here because the dataset is imbalanced (~69/31): a plain random
  split could dump most spam examples into one fold, making that fold's
  metrics unreliable.
- **Multinomial Naive Bayes** — a simple probabilistic classifier that
  assumes word occurrences are conditionally independent given the class.
  It's the traditional textbook baseline for spam filtering (this
  independence assumption is technically wrong but works surprisingly well
  for text classification).
- **Logistic Regression** — a linear model that learns a weight per feature
  and outputs a probability via the sigmoid function. Used with
  `class_weight="balanced"` to compensate for the ham/spam imbalance.
- **XGBoost (gradient-boosted decision trees)** — builds an ensemble of
  shallow decision trees sequentially, where each new tree corrects the
  errors of the ones before it. Handles a mix of sparse (TF-IDF) and dense
  (structural) features better than a linear model because tree splits don't
  require features to be on comparable scales.
- **StandardScaler** — rescales the structural features (which have very
  different ranges — e.g. a 0–1 ratio vs. a raw character count) to zero
  mean and unit variance before combining them with TF-IDF, so no single
  feature dominates purely due to scale.

## Results

Stratified 5-fold CV (`data/processed/baseline_results.csv`):

| Model | Features | Spam Precision | Ham FPR | Spam F1 | Accuracy |
|---|---|---|---|---|---|
| Multinomial NB | TF-IDF | 0.962 | 1.7% | 0.950 | 0.969 |
| Logistic Regression | TF-IDF | 0.960 | 1.8% | 0.960 | 0.975 |
| Logistic Regression | TF-IDF + handcrafted | 0.954 | 2.1% | 0.957 | 0.973 |
| **XGBoost** | **TF-IDF + handcrafted** | **0.977** | **1.0%** | **0.974** | **0.984** |

**XGBoost wins** on both primary metrics simultaneously — the highest spam
precision *and* the lowest false-positive rate, which usually trade off
against each other. Naive Bayes remains a strong, cheap baseline. Notably,
adding handcrafted features to plain Logistic Regression made it *worse*
(sparse TF-IDF + a handful of unscaled-scale dense features don't mix well in
a linear model) — XGBoost's tree splits handle the mixed feature space
better.

## Error Analysis

Full write-up: [`reports/error_analysis.md`](reports/error_analysis.md).
Confusion matrices and ROC/PR curves: `reports/figures/`.

On the held-out test set, XGBoost had the fewest total errors (21, vs. 29–43
for the other models). The dominant confusion pattern in both directions is
**bulk/promotional email**, not "obvious spam vs. obvious ham":

- **False positives** (legit mail flagged as spam) were almost entirely
  newsletter/marketing email from real companies (CNET, Netscape, Red Hat,
  O'Reilly) — they share spam's surface signals (many links, "FREE",
  unsubscribe boilerplate) despite being legitimate opt-in mail.
- **False negatives** (spam missed) split into two patterns: newsletter-style
  spam written in calm, structured prose with no urgency language, and
  casual first-person dating/loan spam that doesn't match typical spam
  templates.

This is a well-known real-world hard case for spam filters, not an artifact
specific to this dataset.

## Project Structure

```
spam-classifier/
├── data/
│   ├── raw/              # downloaded .eml files (not committed)
│   └── processed/        # parsed parquet + feature matrices (not committed)
├── src/
│   ├── parse_emails.py       # raw .eml -> structured parquet
│   ├── features.py           # TF-IDF + structural feature engineering
│   ├── train_classical.py    # cross-validated baseline comparison
│   ├── train_final.py        # fit + persist final models for serving
│   ├── evaluate.py           # confusion matrices, ROC/PR, error analysis
│   └── api/
│       └── main.py           # FastAPI service
├── models/                # saved model/vectorizer artifacts (not committed)
├── notebooks/
│   └── eda.ipynb
├── reports/
│   ├── error_analysis.md
│   ├── misclassified_examples.csv
│   └── figures/
├── logs/
│   └── requests.log       # API request log (model, prediction, confidence, latency)
├── requirements.txt
└── README.md
```

### API usage

```bash
# Health check
curl http://127.0.0.1:8000/health

# Classify raw text (default model: xgboost)
curl -X POST "http://127.0.0.1:8000/classify" -F "text=Congratulations! You've won a free prize, click here now!"

# Classify an .eml file with a specific model
curl -X POST "http://127.0.0.1:8000/classify?model=nb" -F "file=@path/to/email.eml"
```

Available models: `nb`, `logreg_tfidf`, `logreg_combined`, `xgboost` (default).
