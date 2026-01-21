import pickle
import re

# clean text same as training
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text

# load saved files
model = pickle.load(open("models/intent_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf.pkl", "rb"))

def predict_intent(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)
    return prediction[0]

# test examples
if __name__ == "__main__":
    while True:
        user_input = input("Enter a sentence (or type 'exit'): ")
        if user_input.lower() == "exit":
            break
        print("Predicted Intent:", predict_intent(user_input))
