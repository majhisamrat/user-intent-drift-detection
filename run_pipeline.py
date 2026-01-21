
import subprocess
import json
from pathlib import Path

DRIFT_RESULT_PATH = Path("metrics/drift_result.json")


def run_step(command, step_name):
    print(f"\n Running step: {step_name}")
    result = subprocess.run(
        command,
        shell=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"{step_name} failed")


def main():
    # Preprocess
    run_step("python src/preprocess.py", "Preprocessing")

    # Generate embeddings
    run_step("python src/embed.py", "Embedding generation")

    # Drift detection
    run_step("python src/drift_detect.py", "Drift detection")

    # Decide retraining
    if not DRIFT_RESULT_PATH.exists():
        raise FileNotFoundError("Drift result file not found")

    with open(DRIFT_RESULT_PATH, "r") as f:
        drift_data = json.load(f)

    if drift_data.get("drift_detected", False):
        print("\n Drift detected → retraining model")
        run_step("python src/train.py", "Model training")
    else:
        print("\n No drift detected → skipping training")


if __name__ == "__main__":
    main()
