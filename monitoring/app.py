import streamlit as st
import requests
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# CONFIG

st.set_page_config(
    page_title="Intent Monitoring Dashboard",
    layout="wide"
)

API_BASE = "https://user-intent-api.onrender.com"
API_URL = f"{API_BASE}/predict"
LOCAL_API_URL = "http://localhost:8000/predict"

# AUTO REFRESH

refresh = st.sidebar.checkbox("🔄 Live Monitoring", value=True)

if refresh:
    st_autorefresh(interval=5000, key="refresh")

# FETCH DATA FROM API

def fetch_metrics():
    try:
        return requests.get(f"{API_BASE}/metrics", timeout=10).json()
    except:
        return None


def fetch_logs():
    try:
        data = requests.get(f"{API_BASE}/monitor/logs", timeout=10).json()
        return pd.DataFrame(data) if data else pd.DataFrame()
    except:
        return pd.DataFrame()
    
# UI START

st.title("🧠 User Intent Monitoring Dashboard")

# TRAINING METRICS

st.header("📊 Training Metrics")

metrics = fetch_metrics()

if metrics and "accuracy" in metrics:
    c1, c2, c3 = st.columns(3)

    c1.metric("Accuracy", round(metrics.get("accuracy", 0), 4))
    c2.metric("F1 Score", round(metrics.get("f1_score", 0), 4))
    c3.metric("Model", metrics.get("model", "N/A"))
else:
    st.warning("Metrics not available from API.")

# LOAD LOG DATA

df_logs = fetch_logs()

# PREDICTION OVERVIEW

st.header("📈 Prediction Overview")

if not df_logs.empty:

    total = len(df_logs)
    rejected = df_logs["is_rejected"].sum()
    accepted = total - rejected
    rejection_rate = (rejected / total) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Predictions", total)
    col2.metric("Accepted", accepted)
    col3.metric("Rejected", rejected)
    col4.metric("Rejection Rate", f"{round(rejection_rate, 2)}%")

else:
    st.info("No prediction data available yet.")

# CONFIDENCE ANALYSIS

st.header("📉 Confidence Analysis")

if not df_logs.empty and "confidence" in df_logs.columns:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confidence Over Time")
        st.line_chart(df_logs["confidence"])

    with col2:
        st.subheader("Confidence Histogram")
        st.bar_chart(
            df_logs["confidence"]
            .value_counts(bins=20)
            .sort_index()
        )

else:
    st.info("No confidence data available.")

# INTENT DISTRIBUTION

st.header("🥧 Intent Distribution")

if not df_logs.empty and "predicted_intent" in df_logs.columns:

    intent_counts = df_logs["predicted_intent"].value_counts()
    st.bar_chart(intent_counts)

else:
    st.info("No intent data available.")

# LATEST PREDICTION

st.header("⚡ Latest Prediction")

if not df_logs.empty:
    last = df_logs.iloc[-1]

    c1, c2, c3 = st.columns(3)

    c1.metric("Intent", last.get("predicted_intent", "N/A"))
    c2.metric("Confidence", round(last.get("confidence", 0), 4))
    c3.metric("Rejected", last.get("is_rejected", False))
else:
    st.info("No predictions logged yet.")

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