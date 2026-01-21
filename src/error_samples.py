import pickle
import pandas as pd
from preprocess import preprocess
from vectorize import x_test_vec

model = pickle.load(open("models/intent_model.pkl", "rb"))

_, test_df = preprocess()

test_df["predicted_intent"] = model.predict(x_test_vec)

errors = test_df[test_df["intent"] != test_df["predicted_intent"]]

errors.to_csv("data/raw/misclassified_samples.csv", index=False)

print("Saved misclassified samples:", len(errors))
