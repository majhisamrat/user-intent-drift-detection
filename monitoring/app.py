import numpy as np
import streamlit as st
import json
import os
import requests
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# CONFIG

st.set_page_config(
    page_title="Intent Monitoring Dashboard",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

TRAINING_METRICS_PATH = os.path.join(ROOT_DIR, "metrics", "training_metrics.json")
PREDICTIONS_LOG_PATH = os.path.join(ROOT_DIR, "logs", "predictions.jsonl")

API_URL = "https://user-intent-api.onrender.com/predict"
LOCAL_API_URL = "http://localhost:8000/predict"

CONFIDENCE_THRESHOLD = 0.108

# AUTO REFRESH

refresh = st.sidebar.checkbox("🔄 Live Monitoring", value=True)

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

# UI START

st.title("🧠 User Intent Monitoring Dashboard")

# TRAINING METRICS

st.header("📊 Training Metrics")

metrics = load_json(TRAINING_METRICS_PATH)

if metrics:
    c1, c2, c3 = st.columns(3)

    c1.metric("Accuracy", round(metrics.get("accuracy", 0), 4))
    c2.metric("F1 Score", round(metrics.get("f1_score", 0), 4))
    c3.metric("Loss", round(metrics.get("loss", 0), 4))
else:
    st.warning("training_metrics.json not found")

# LOAD LOG DATA

df_logs = load_predictions(PREDICTIONS_LOG_PATH)

# LOW CONFIDENCE MONITORING

st.header("🚫 Low Confidence Monitoring")

if not df_logs.empty and "confidence" in df_logs.columns:

    confidence_series = df_logs["confidence"]

    rejected_count = (confidence_series < CONFIDENCE_THRESHOLD).sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Logged Predictions", len(df_logs))
    col2.metric("Total Rejected Predictions", rejected_count)
    col3.metric(
        "Rejection Rate",
        f"{round((rejected_count / len(df_logs)) * 100, 2)}%"
    )

else:
    st.info("No prediction data available yet.")

# CONFIDENCE DISTRIBUTION

st.header("📉 Confidence Distribution")

if not df_logs.empty and "confidence" in df_logs.columns:

    confidence_series = df_logs["confidence"]

    col1, col2 = st.columns(2)

    # ---- LINE CHART ----
    with col1:
        st.subheader("Confidence Over Time")
        st.line_chart(confidence_series, height=300)

    # ---- HISTOGRAM ----
    with col2:
        st.subheader("Confidence Histogram")

        hist_df = pd.DataFrame({
            "confidence": confidence_series
        })

        st.bar_chart(
            hist_df["confidence"].value_counts(
                bins=20
            ).sort_index(),
            height=300
        )

else:
    st.info("No confidence data available.")

# LATEST PREDICTION


st.header("⚡ Latest Prediction")

if not df_logs.empty:
    last = df_logs.iloc[-1]

    c1, c2 = st.columns(2)

    c1.metric("Intent", last.get("predicted_intent", "N/A"))
    c2.metric("Confidence", round(last.get("confidence", 0), 4))

# RECENT TABLE

st.header("🗂 Recent Predictions")

if not df_logs.empty:
    st.dataframe(df_logs.tail(50), use_container_width=True)
else:
    st.info("No predictions logged yet.")

# API TESTER

st.divider()
st.header("🚀 Test FastAPI /predict")

use_local = st.toggle("Use localhost API", value=False)

user_text = st.text_input("Enter message")

if st.button("Predict Intent"):
    if user_text:
        try:
            url = LOCAL_API_URL if use_local else API_URL
            res = requests.post(url, json={"text": user_text})
            st.success(res.json())
        except Exception as e:
            st.error(str(e))