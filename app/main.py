from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import re
import numpy as np
import json
import os
import pandas as pd
from datetime import datetime


# FastAPI App

app = FastAPI(
    title="User Intent Classification API",
    description="Predicts user intent with confidence and logs predictions",
    version="1.0.0"
)

# Load model artifacts


MODEL_PATH = "models/intent_model_v1.pkl"
VECTORIZER_PATH = "models/tfidf_v1.pkl"

model = None
vectorizer = None

if os.path.exists(MODEL_PATH):
    model = pickle.load(open(MODEL_PATH, "rb"))

if os.path.exists(VECTORIZER_PATH):
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))


# Pydantic Schemas

class IntentRequest(BaseModel):
    text: str = Field(..., example="How can I change my pin?")

class IntentResponse(BaseModel):
    intent: str
    confidence: float

# Utils

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text

               
# Logging

def log_prediction(text, intent, confidence):

    log_dir = "/logs"
    os.makedirs(log_dir, exist_ok=True)

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "text": text,
        "predicted_intent": intent,
        "confidence": confidence
    }

    log_path = os.path.join(log_dir, "predictions.jsonl")

    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


# Routes


@app.get("/")
def home():
    return {"message": "Intent Classification API is running"}

# Predict Endpoint

@app.post("/predict", response_model=IntentResponse)
def predict_intent(request: IntentRequest):

    if model is None or vectorizer is None:
        return {"intent": "model_not_loaded", "confidence": 0.0}

    cleaned_text = clean_text(request.text)
    vec = vectorizer.transform([cleaned_text])

    probs = model.predict_proba(vec)[0]
    best_index = int(np.argmax(probs))

    best_intent = model.classes_[best_index]
    confidence = round(float(probs[best_index]), 3)

    # log prediction
    log_prediction(request.text, best_intent, confidence)

    return {
        "intent": best_intent,
        "confidence": confidence
    }


# Model Info

@app.get("/model-info")
def model_info():
    return {
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "model_type": str(type(model)) if model else None
    }


# Training Metrics

@app.get("/metrics")
def metrics():

    metrics_path = "metrics/training_metrics.json"

    if not os.path.exists(metrics_path):
        return {"error": "metrics file not found"}

    with open(metrics_path, "r") as f:
        data = json.load(f)

    return data


# Drift Status

@app.get("/drift-status")
def drift_status():

    baseline_path = "models/baseline_intent_distribution.json"
    logs_path = "logs/predictions.jsonl"

    if not os.path.exists(baseline_path):
        return {"error": "baseline not found"}

    if not os.path.exists(logs_path):
        return {"status": "no_predictions_yet"}

    logs = pd.read_json(logs_path, lines=True)

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
        "num_predictions": len(logs)
    }
