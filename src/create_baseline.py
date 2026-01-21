import pandas as pd
from preprocess import preprocess
import json
import os

# Load processed training data
train_df, _ = preprocess()

# Intent distribution baseline
baseline_dist = (
    train_df["intent"]
    .value_counts(normalize=True)
    .to_dict()
)

# Save path inside Airflow container
BASELINE_PATH = "/opt/airflow/models/baseline_intent_distribution.json"
os.makedirs("/opt/airflow/models", exist_ok=True)

# Save baseline
with open(BASELINE_PATH, "w") as f:
    json.dump(baseline_dist, f, indent=4)

print("Baseline intent distribution saved")
