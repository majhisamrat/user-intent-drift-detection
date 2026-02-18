import streamlit as st
import json
import os
import requests
import pandas as pd
import numpy as np
from collections import Counter
from streamlit_autorefresh import st_autorefresh

# CONFIG

st.set_page_config(page_title="Intent Drift Monitor", layout="wide")

TRAINING_METRICS_PATH = "/metrics/training_metrics.json"
BASELINE_DIST_PATH = "/models/baseline_intent_distribution.json"
PREDICTIONS_LOG_PATH = "/logs/predictions.jsonl"

API_URL = "http://intent_api:8000/predict"
LOCAL_API_URL = "http://localhost:8000/predict"

# AUTO REFRESH (NO LOOP)

refresh = st.sidebar.checkbox("Live Monitoring (Auto Refresh)", value=True)

if refresh:
    st_autorefresh(interval=5000, key="datarefresh")


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


def calculate_distribution(intents):
    counter = Counter(intents)
    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()} if total > 0 else {}


def kl_divergence(p, q):
    score = 0.0
    for key in q:
        p_val = p.get(key, 1e-6)
        q_val = q.get(key, 1e-6)
        score += p_val * np.log(p_val / q_val)
    return float(score)

# UI

st.title("🧠 User Intent Drift Monitoring Dashboard")

#  TRAINING METRICS 
st.header("📊 Training Metrics")

metrics = load_json(TRAINING_METRICS_PATH)

if metrics:
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", metrics.get("accuracy", "N/A"))
    c2.metric("F1 Score", metrics.get("f1_score", "N/A"))
    c3.metric("Loss", metrics.get("loss", "N/A"))
else:
    st.warning("training_metrics.json not found")

# ---------- LOAD DATA ----------
df_logs = load_predictions(PREDICTIONS_LOG_PATH)
baseline_dist = load_json(BASELINE_DIST_PATH)

# DISTRIBUTION 
st.header("📈 Intent Distribution")

if not df_logs.empty and "predicted_intent" in df_logs.columns:

    current_dist = calculate_distribution(df_logs["predicted_intent"])

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Current Distribution")
        st.bar_chart(pd.DataFrame.from_dict(current_dist, orient="index"))

    if baseline_dist:
        with colB:
            st.subheader("Baseline Distribution")
            st.bar_chart(pd.DataFrame.from_dict(baseline_dist, orient="index"))

        drift = kl_divergence(current_dist, baseline_dist)
        st.metric("KL Drift Score", round(drift, 4))

        if drift < 0.15:
            st.success("🟢 Stable")
        elif drift < 0.3:
            st.warning("🟡 Drift Warning")
        else:
            st.error("🔴 Drift Detected")

else:
    st.info("No prediction logs available yet.")

#  CONFIDENCE 
st.header("📉 Confidence Monitoring")

if not df_logs.empty and "confidence" in df_logs.columns:
    st.line_chart(df_logs["confidence"])

# LATEST 
st.header("⚡ Latest Prediction")

if not df_logs.empty:
    last = df_logs.iloc[-1]
    c1, c2 = st.columns(2)
    c1.metric("Intent", last["predicted_intent"])
    c2.metric("Confidence", last["confidence"])

# TABLE 
st.header("🗂 Prediction Logs")

if not df_logs.empty:
    st.dataframe(df_logs.tail(50), use_container_width=True)

# FASTAPI TESTER (STATIC)

st.divider()
st.header("🚀 Test FastAPI /predict")

use_local = st.toggle(
    "Use localhost API instead of docker service",
    value=False,
    key="predict_toggle_static"
)

user_text = st.text_input(
    "Enter message",
    key="predict_input_static"
)

if st.button("Predict Intent", key="predict_button_static"):
    if user_text:
        try:
            url = LOCAL_API_URL if use_local else API_URL
            res = requests.post(url, json={"text": user_text})
            st.success(res.json())
        except Exception as e:
            st.error(str(e))
