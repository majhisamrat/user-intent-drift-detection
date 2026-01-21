import os
import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import preprocess

MODEL_DIR = "/opt/airflow/models"

def main():
    # Load processed data
    _, test_df = preprocess()

    x_test = test_df["text"]
    y_test = test_df["intent"]

    # Load vectorizer
    with open(os.path.join(MODEL_DIR, "tfidf_v1.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    # Load model
    with open(os.path.join(MODEL_DIR, "intent_model_v1.pkl"), "rb") as f:
        model = pickle.load(f)

    # Vectorize test data
    x_test_vec = vectorizer.transform(x_test)

    # Predict
    y_pred = model.predict(x_test_vec)

    # Evaluation metrics
    print("\nCONFUSION MATRIX ")
    print(confusion_matrix(y_test, y_pred))

    print("\nCLASSIFICATION REPORT ")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
