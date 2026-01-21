import os
import pandas as pd

LOG_PATH = "/opt/airflow/logs/predictions.jsonl"

def main():
    # Case 1: file does not exist
    if not os.path.exists(LOG_PATH):
        print("No prediction logs found. Skipping confidence drift check.")
        return

    # Case 2: file exists but empty
    if os.stat(LOG_PATH).st_size == 0:
        print("Prediction log file is empty. Skipping confidence drift check.")
        return

    # Load logs safely
    logs = pd.read_json(LOG_PATH, lines=True)

    if logs.empty:
        print("No valid prediction records found.")
        return

    avg_confidence = logs["confidence"].mean()

    print(f"Average prediction confidence: {avg_confidence:.3f}")

    # Simple drift rule
    if avg_confidence < 0.6:
        print("⚠️ Confidence drift detected!")
    else:
        print("✅ No confidence drift detected")

if __name__ == "__main__":
    main()
