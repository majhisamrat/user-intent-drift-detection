import json
import os
import pandas as pd
from preprocess import preprocess

MODEL_DIR = "/opt/airflow/models"
BASELINE_PATH = os.path.join(MODEL_DIR, "baseline_intent_distribution.json")

def main():
    # Ensure model directory exists
    if not os.path.exists(BASELINE_PATH):
        raise FileNotFoundError(f"Baseline file not found at {BASELINE_PATH}")

    # Load baseline
    with open(BASELINE_PATH, "r") as f:
        baseline = json.load(f)

    # Load current data
    train_df, _ = preprocess()

    current_dist = (
        train_df["intent"]
        .value_counts(normalize=True)
        .to_dict()
    )

    # Simple drift check
    drift_detected = False
    for intent, baseline_prob in baseline.items():
        current_prob = current_dist.get(intent, 0.0)
        if abs(current_prob - baseline_prob) > 0.1:
            drift_detected = True
            print(
                f"Drift detected for intent '{intent}': "
                f"baseline={baseline_prob:.3f}, current={current_prob:.3f}"
            )

    if not drift_detected:
        print("No significant intent drift detected")

if __name__ == "__main__":
    main()
