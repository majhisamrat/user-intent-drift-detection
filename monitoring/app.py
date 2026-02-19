import numpy as np
import streamlit as st
import json
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

# CONFIG

st.set_page_config(page_title="Intent Monitoring Dashboard", layout="wide")

TRAINING_METRICS_PATH = "/metrics/training_metrics.json"
PREDICTIONS_LOG_PATH = "/logs/predictions.jsonl"

API_URL = "http://intent_api:8000/predict"
LOCAL_API_URL = "http://localhost:8000/predict"

CONFIDENCE_THRESHOLD = 0.108

# AUTO REFRESH

refresh = st.sidebar.checkbox("Live Monitoring (Auto Refresh)", value=True)

if refresh:
    st_autorefresh(interval=5000, key="refresh")

# HELPERS

def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def load_predictions(path):
    if not os.path.exists(path):
        return pd.DataFrame()

    rows = []
    with open(path, "r") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except:
                pass

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# UI

st.title("🧠 User Intent Monitoring Dashboard")

# TRAINING METRICS

st.header("📊 Training Metrics")

metrics = load_json(TRAINING_METRICS_PATH)

if metrics:
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", metrics.get("accuracy", "N/A"))
    c2.metric("F1 Score", metrics.get("f1_score", "N/A"))
    c3.metric("Loss", metrics.get("loss", "N/A"))
else:
    st.warning("training_metrics.json not found")

# LOAD LOG DATA

df_logs = load_predictions(PREDICTIONS_LOG_PATH)

# REJECTED COUNTER

st.header("🚫 Low Confidence Monitoring")

if not df_logs.empty and "confidence" in df_logs.columns:

    rejected_df = df_logs[df_logs["confidence"] < CONFIDENCE_THRESHOLD]
    total_rejected = len(rejected_df)

    st.metric("Total Rejected Predictions", total_rejected)

else:
    st.info("No prediction data available yet.")

# CONFIDENCE DISTRIBUTION 

st.header("📉 Confidence Distribution")

if not df_logs.empty and "confidence" in df_logs.columns:

    confidence_series = df_logs["confidence"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confidence Over Time")
        st.line_chart(confidence_series)

    with col2:
        st.subheader("Confidence Histogram")

        hist_values = np.histogram(confidence_series, bins=20)[0]
        st.bar_chart(hist_values)

    # Threshold line info
    threshold = 0.169
    rejected_count = (confidence_series < threshold).sum()

    st.metric("Total Rejected Predictions", rejected_count)

else:
    st.info("No confidence data available.")


# LATEST PREDICTION


st.header("⚡ Latest Prediction")

if not df_logs.empty:
    last = df_logs.iloc[-1]
    c1, c2 = st.columns(2)
    c1.metric("Intent", last.get("predicted_intent", "N/A"))
    c2.metric("Confidence", last.get("confidence", "N/A"))

# PREDICTION TABLE

st.header("🗂 Recent Predictions")

if not df_logs.empty:
    st.dataframe(df_logs.tail(50), use_container_width=True)
else:
    st.info("No predictions logged yet.")

# FASTAPI TESTER


st.divider()
st.header("🚀 Test FastAPI /predict")

use_local = st.toggle(
    "Use localhost API instead of docker service",
    value=False,
    key="predict_toggle"
)

user_text = st.text_input("Enter message")

if st.button("Predict Intent"):
    if user_text:
        try:
            url = LOCAL_API_URL if use_local else API_URL
            res = requests.post(url, json={"text": user_text})
            st.success(res.json())
        except Exception as e:
            st.error(str(e))
