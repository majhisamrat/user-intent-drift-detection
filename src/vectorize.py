from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from preprocess import preprocess
import pickle
import os


# load cleaned data
train_df, test_df = preprocess()

# train-validation split
x_train, x_val, y_train, y_val = train_test_split(
    train_df["clean_text"],
    train_df["intent"],
    test_size=0.1,
    random_state=42,
    stratify=train_df["intent"]
)

# TF-IDF vectorizer

vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),       
    max_features=30000,        
    min_df=2,                  
    max_df=0.9,                
    sublinear_tf=True        
)
x_train_vec = vectorizer.fit_transform(x_train)
x_val_vec = vectorizer.transform(x_val)
x_test_vec = vectorizer.transform(test_df["clean_text"])

# save vectorizer

MODEL_DIR = "/opt/airflow/models"
os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, "tfidf_v1.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print("Vectorization completed")
print("Train shape:", x_train_vec.shape)
print("Validation shape:", x_val_vec.shape)
print("Test shape:", x_test_vec.shape)

# expose variables for training
x_train_vec = x_train_vec
x_val_vec = x_val_vec
y_train = y_train
y_val = y_val

