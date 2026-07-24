import os
import sys
import joblib
import pandas as pd

# -------------------------------------------------
# Add Project Root
# -------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.preprocessing import clean_text
from utils.tokenizer import tokenize_and_remove_stopwords


def prepare_text(text):
    """
    Complete preprocessing pipeline.
    """
    text = clean_text(text)
    text = tokenize_and_remove_stopwords(text)
    return text


# -------------------------------------------------
# Load Model and Vectorizer
# -------------------------------------------------

print("=" * 60)
print("Batch Fake News Prediction")
print("=" * 60)

model = joblib.load("models/logistic_regression.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

# -------------------------------------------------
# Read Input CSV
# -------------------------------------------------

input_file = input("Enter CSV file path: ").strip()

if not os.path.exists(input_file):
    print(f"\n❌ File not found: {input_file}")
    sys.exit()

df = pd.read_csv(input_file)

if "text" not in df.columns:
    print("\n❌ CSV must contain a 'text' column.")
    sys.exit()

print(f"\nLoaded {len(df)} articles.")

# -------------------------------------------------
# Preprocess
# -------------------------------------------------

print("Preprocessing articles...")

df["processed_text"] = df["text"].apply(prepare_text)

# -------------------------------------------------
# Vectorize
# -------------------------------------------------

features = vectorizer.transform(df["processed_text"])

# -------------------------------------------------
# Predict
# -------------------------------------------------

predictions = model.predict(features)
probabilities = model.predict_proba(features)

df["prediction"] = [
    "FAKE NEWS" if pred == 0 else "REAL NEWS"
    for pred in predictions
]

df["fake_probability"] = (probabilities[:, 0] * 100).round(2)
df["real_probability"] = (probabilities[:, 1] * 100).round(2)

# -------------------------------------------------
# Save Output
# -------------------------------------------------

output_file = "predictions.csv"

df.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("Batch Prediction Completed!")
print("=" * 60)
print(f"Total Articles : {len(df)}")
print(f"Results Saved  : {output_file}")

print("\nSample Predictions:")
print(
    df[
        [
            "prediction",
            "fake_probability",
            "real_probability",
        ]
    ].head()
)

print("=" * 60)