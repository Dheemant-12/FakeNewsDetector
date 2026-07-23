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

    # -------------------------------
    # Preprocess
    # -------------------------------

    processed = prepare_text(article)

    print("\nProcessed Text:")
    print(processed)

    # -------------------------------
    # Vectorize
    # -------------------------------

    features = vectorizer.transform([processed])

    print(f"\nFeature Shape      : {features.shape}")
    print(f"Non-zero Features  : {features.nnz}")

    # -------------------------------
    # Prediction
    # -------------------------------

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    fake_probability = probabilities[0] * 100
    real_probability = probabilities[1] * 100

    # -------------------------------
    # Output
    # -------------------------------

    print("\n" + "=" * 60)

    if prediction == 0:
        print("Prediction : FAKE NEWS")
    else:
        print("Prediction : REAL NEWS")

    print(f"Fake Probability : {fake_probability:.2f}%")
    print(f"Real Probability : {real_probability:.2f}%")

    print("=" * 60)