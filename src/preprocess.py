import pandas as pd
import re

def load_data():
    train_df = pd.read_csv("/opt/airflow/data/raw/train.csv")
    test_df = pd.read_csv("/opt/airflow/data/raw/test.csv")


    # rename category → intent (direct intent names)
    train_df.rename(columns={"category": "intent"}, inplace=True)
    test_df.rename(columns={"category": "intent"}, inplace=True)

    return train_df, test_df


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text


def preprocess():
    train_df, test_df = load_data()

    train_df["clean_text"] = train_df["text"].apply(clean_text)
    test_df["clean_text"] = test_df["text"].apply(clean_text)

    return train_df, test_df
