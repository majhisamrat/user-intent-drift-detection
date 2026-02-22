from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import re
import numpy as np
import json
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

# CONFIG

CONFIDENCE_THRESHOLD = 0.169

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "metrics"

LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "predictions.jsonl"
MODEL_PATH = MODEL_DIR / "intent_model_v1.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_v1.pkl"

# FASTAPI APP

app = FastAPI(
    title="User Intent Classification API",
    description="Predicts user intent with full prediction logging & monitoring",
    version="3.0.0"
)

# LOAD MODEL

model = None
vectorizer = None

if MODEL_PATH.exists():
    model = pickle.load(open(MODEL_PATH, "rb"))

if VECTORIZER_PATH.exists():
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))

# SCHEMAS

class IntentRequest(BaseModel):
    text: str = Field(..., example="How can I change my pin?")

class IntentResponse(BaseModel):
    intent: str
    confidence: float

# UTILITIES
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text


def log_prediction(text, predicted_intent, confidence, is_rejected):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "text": text,
        "predicted_intent": predicted_intent,
        "confidence": confidence,
        "is_rejected": is_rejected
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# ROUTES

@app.get("/")
def home():
    return {"message": "Intent Classification API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}

# PREDICT

@app.post("/predict", response_model=IntentResponse)
def predict_intent(request: IntentRequest):

    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    text = clean_text(request.text)

    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]

    predicted_class = model.classes_[np.argmax(probs)]
    confidence = float(np.max(probs))

    # Determine rejection
    is_rejected = confidence < CONFIDENCE_THRESHOLD

    if is_rejected:
        predicted_class = "not_identified"

    # Log ALL predictions
    log_prediction(request.text, predicted_class, confidence, is_rejected)

    return {
        "intent": predicted_class,
        "confidence": confidence
    }

# MONITORING ENDPOINTS

@app.get("/monitor/logs")
def get_logs():

    if not LOG_FILE.exists():
        return []

    data = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass

    return data


@app.get("/monitor/stats")
def get_stats():

    if not LOG_FILE.exists():
        return {
            "total_predictions": 0,
            "total_rejected": 0,
            "average_confidence": 0
        }

    df = pd.read_json(LOG_FILE, lines=True)

    if df.empty:
        return {
            "total_predictions": 0,
            "total_rejected": 0,
            "average_confidence": 0
        }

    return {
        "total_predictions": len(df),
        "total_rejected": int(df["is_rejected"].sum()),
        "average_confidence": round(float(df["confidence"].mean()), 4)
    }

# TRAINING METRICS

@app.get("/metrics")
def metrics():

    metrics_path = METRICS_DIR / "training_metrics.json"

    if not metrics_path.exists():
        return {"error": "metrics file not found"}

    return json.load(open(metrics_path))

# DRIFT STATUS

@app.get("/drift-status")
def drift_status():

    baseline_path = MODEL_DIR / "baseline_intent_distribution.json"

    if not baseline_path.exists():
        return {"error": "baseline not found"}

    if not LOG_FILE.exists():
        return {"status": "no_predictions_yet"}

    logs = pd.read_json(LOG_FILE, lines=True)

    if logs.empty:
        return {"status": "empty_logs"}

    baseline = json.load(open(baseline_path))

    current_distribution = (
        logs["predicted_intent"]
        .value_counts(normalize=True)
        .to_dict()
    )

    return {
        "baseline_distribution": baseline,
        "current_distribution": current_distribution,
        "total_predictions": len(logs),
        "rejection_rate": round(float(logs["is_rejected"].mean()), 4)
    }


# MODEL INFO

@app.get("/model-info")
def model_info():
    return {
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "model_type": str(type(model)) if model else None
    }