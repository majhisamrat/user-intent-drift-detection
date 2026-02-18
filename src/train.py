
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from vectorize import (
    x_train_vec,
    x_val_vec,
    y_train,
    y_val
)
import pickle
import os

os.makedirs("models", exist_ok=True)

# initialize model
model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)

# train
model.fit(x_train_vec, y_train)

# predict
y_pred = model.predict(x_val_vec)

# evaluation
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_val, y_pred))

# save model
MODEL_DIR = "/opt/airflow/models"
os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, "intent_model_v1.pkl"), "wb") as f:
    pickle.dump(model, f)

print("Model training completed and saved")
