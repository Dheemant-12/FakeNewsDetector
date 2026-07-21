import os
import sys
import joblib

# -------------------------------------------------
# Add project root
# -------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.preprocessing import clean_text
from utils.tokenizer import tokenize_and_remove_stopwords


def prepare_text(text):
    text = clean_text(text)
    text = tokenize_and_remove_stopwords(text)
    return text


print("=" * 60)
print("Fake News Detector")
print("=" * 60)

model = joblib.load("models/logistic_regression.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

while True:

    print("\nEnter a news article.")
    print("Type 'exit' to quit.\n")

    article = input("News: ")

    if article.lower() == "exit":
        print("\nGoodbye!")
        break

    processed = prepare_text(article)

    features = vectorizer.transform([processed])

    prediction = model.predict(features)[0]

    confidence = model.predict_proba(features).max() * 100

    print("\n" + "=" * 60)

    if prediction == 0:
        print("Prediction : FAKE NEWS")
    else:
        print("Prediction : REAL NEWS")

    print(f"Confidence : {confidence:.2f}%")

    print("=" * 60)