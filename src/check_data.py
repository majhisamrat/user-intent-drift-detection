from preprocess import preprocess

train_df, test_df = preprocess()

print("Train samples:", len(train_df))
print("Test samples:", len(test_df))
print("Unique intents:", train_df["intent"].nunique())
print(train_df[["text", "intent"]].head())
