"""FastAPI service for the email spam classifier.

Run with:
    uvicorn api.main:app --reload --app-dir src

Endpoints:
    GET  /health                          -> service liveness check
    POST /classify?model=xgboost          -> classify raw text or an .eml upload
"""

import email
import email.policy
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from scipy import sparse

sys.path.append(str(Path(__file__).resolve().parent.parent))
from features import build_structural_features, clean_text, ensure_nltk_data  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "requests.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("spam_classifier")

app = FastAPI(title="Email Spam Classifier")

MODEL_REGISTRY = {
    "nb": {"file": "nb_tfidf.joblib", "features": "tfidf"},
    "logreg_tfidf": {"file": "logreg_tfidf.joblib", "features": "tfidf"},
    "logreg_combined": {"file": "logreg_combined.joblib", "features": "combined"},
    "xgboost": {"file": "xgboost_combined.joblib", "features": "combined"},
}
DEFAULT_MODEL = "xgboost"

_artifacts = {}


@app.on_event("startup")
def load_artifacts():
    ensure_nltk_data()

    _artifacts["vectorizer"] = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    _artifacts["scaler"] = joblib.load(MODELS_DIR / "structural_scaler.joblib")
    _artifacts["structural_columns"] = joblib.load(MODELS_DIR / "structural_columns.joblib")

    _artifacts["models"] = {}
    for name, spec in MODEL_REGISTRY.items():
        path = MODELS_DIR / spec["file"]
        if path.exists():
            _artifacts["models"][name] = joblib.load(path)

    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    _artifacts["stop_words"] = set(stopwords.words("english"))
    _artifacts["lemmatizer"] = WordNetLemmatizer()

    logger.info(f"Loaded models: {list(_artifacts['models'].keys())}")


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    model_used: str


def parse_eml_bytes(raw_bytes: bytes) -> dict:
    msg = email.message_from_bytes(raw_bytes, policy=email.policy.default)

    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_content()
                break
        else:
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    body = part.get_content()
                    break
    else:
        body = msg.get_content()

    return {
        "from": msg.get("From", ""),
        "subject": msg.get("Subject", ""),
        "reply_to": msg.get("Reply-To", ""),
        "received_count": len(msg.get_all("Received", [])),
        "body": body,
    }


def build_feature_row(fields: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "from": fields.get("from", ""),
        "subject": fields.get("subject", ""),
        "reply_to": fields.get("reply_to", ""),
        "received_count": fields.get("received_count", 0),
        "body": fields.get("body", ""),
    }])


def predict(fields: dict, model_name: str) -> ClassifyResponse:
    if model_name not in _artifacts["models"]:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'. Available: {list(_artifacts['models'].keys())}")

    row_df = build_feature_row(fields)

    cleaned = clean_text(row_df["body"].iloc[0], _artifacts["stop_words"], _artifacts["lemmatizer"])
    tfidf_vec = _artifacts["vectorizer"].transform([cleaned])

    features_needed = MODEL_REGISTRY[model_name]["features"]
    if features_needed == "tfidf":
        X = tfidf_vec
    else:
        structural = build_structural_features(row_df)
        structural = structural.reindex(columns=_artifacts["structural_columns"], fill_value=0)
        structural_scaled = _artifacts["scaler"].transform(structural)
        X = sparse.hstack([tfidf_vec, sparse.csr_matrix(structural_scaled)]).tocsr()

    clf = _artifacts["models"][model_name]
    proba = clf.predict_proba(X)[0]
    pred_idx = int(proba.argmax())
    label = "spam" if pred_idx == 1 else "ham"
    confidence = float(proba[pred_idx])

    return ClassifyResponse(label=label, confidence=confidence, model_used=model_name)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(_artifacts.get("models", {}).keys())}


@app.post("/classify", response_model=ClassifyResponse)
async def classify(
    model: str = DEFAULT_MODEL,
    text: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
):
    start = time.time()

    if file is not None:
        raw_bytes = await file.read()
        fields = parse_eml_bytes(raw_bytes)
        source = f"file:{file.filename}"
    elif text is not None and text.strip():
        fields = {"from": "", "subject": "", "reply_to": "", "received_count": 0, "body": text}
        source = "raw_text"
    else:
        raise HTTPException(status_code=400, detail="Provide either 'text' (form field) or 'file' (.eml upload)")

    result = predict(fields, model)

    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        f"source={source} model={result.model_used} label={result.label} "
        f"confidence={result.confidence:.4f} latency_ms={elapsed_ms:.1f}"
    )

    return result
