from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import re
import numpy as np
import json
import os
from datetime import datetime

# App 
app = FastAPI(
    title="User Intent Classification API",
    description="Predicts user intent with confidence and logs predictions",
    version="1.0.0"
)

#  Load model artifacts 
model = pickle.load(open("models/intent_model_v1.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_v1.pkl", "rb"))

# Pydantic Schemas 
class IntentRequest(BaseModel):
    text: str = Field(
        ...,
        example="How can I change my pin?"
    )

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
    os.makedirs("logs", exist_ok=True)

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "text": text,
        "predicted_intent": intent,
        "confidence": confidence
    }

    with open("logs/predictions.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# Routes 
@app.get("/")
def home():
    return {"message": "Intent Classification API is running"}

@app.post("/predict", response_model=IntentResponse)
def predict_intent(request: IntentRequest):
    cleaned_text = clean_text(request.text)
    vec = vectorizer.transform([cleaned_text])

    probs = model.predict_proba(vec)[0]
    best_index = int(np.argmax(probs))

    best_intent = model.classes_[best_index]
    confidence = round(float(probs[best_index]), 3)

    # log prediction (ALWAYS)
    log_prediction(request.text, best_intent, confidence)

    # confidence threshold
    if confidence < 0.5:
        return {
            "intent": "uncertain",
            "confidence": confidence
        }

    return {
        "intent": best_intent,
        "confidence": confidence
    }
