# Error Analysis

Evaluation is based on a single stratified 80/20 train/test split (see `src/evaluate.py`).
Confusion matrices and ROC/PR curves for all four models are in `reports/figures/`.
Full list of sampled misclassified emails: `reports/misclassified_examples.csv`.

## Error counts on the held-out test set

| Model | Features | False Positives (ham→spam) | False Negatives (spam→ham) | Total Errors |
|---|---|---|---|---|
| Multinomial NB | TF-IDF | 15 | 25 | 40 |
| Logistic Regression | TF-IDF | 17 | 12 | 29 |
| Logistic Regression | TF-IDF + handcrafted | 21 | 22 | 43 |
| **XGBoost** | **TF-IDF + handcrafted** | **10** | **11** | **21** |

XGBoost has the fewest errors overall and the best balance between false positives and false
negatives, consistent with the cross-validated results in `data/processed/baseline_results.csv`.

## False positives — legitimate ham misclassified as spam (XGBoost)

Nearly every false positive comes from the `hard_ham` folder and is a **bulk marketing or
newsletter email from a real company**:

- CNET auction alerts ("UBID Auction Alert", "Intel 900MHz... Starting Bid $229")
- Netscape browser release announcement ("Download it now for FREE!")
- Red Hat Network product update
- O'Reilly conference newsletter
- A stock-advice newsletter ("Cash Flow Doesn't Lie...")
- A movie trivia newsletter

**Pattern:** these emails share spam's surface signals — many links, promotional language,
words like "FREE" and "click here," unsubscribe boilerplate — despite being legitimate opt-in
mail. This is exactly why `hard_ham` exists as a corpus category, and it mirrors a real-world
complaint about spam filters: being too aggressive on newsletters.

## False negatives — spam misclassified as ham (XGBoost)

Two distinct sub-patterns emerged:

1. **Newsletter-style spam** (e.g. "TVPredictions.com Newsletter", "Your Membership Exchange",
   an ISP offer) — the mirror image of the false positives above. Written in calm, structured
   newsletter prose with no urgency keywords or heavy caps/exclamation usage, so neither the
   handcrafted urgency features nor obvious spam vocabulary fire.
2. **Dating/webcam and loan spam** written as casual first-person messages
   ("Hi, my name is Kelly, I am an 18 year old...", "Your home refinance loan is approved!")
   rather than typical spam-template language. This evades urgency-keyword detection and reads
   more like a personal message than bulk spam.

## Takeaway

The model's dominant confusion isn't "obvious spam vs. obvious ham" — it's **bulk/promotional
email in general**, in both directions. Newsletter-style content sits right at the ham/spam
boundary regardless of which side it's actually on, which is a well-known hard case for spam
filters in practice, not an artifact specific to this dataset.
