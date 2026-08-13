import os
import sys
import joblib
import pandas as pd


# -------------------------------------------------
# Add Project Root
# -------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# -------------------------------------------------
# Project Imports
# -------------------------------------------------

from utils.preprocessing import clean_text
from utils.tokenizer import tokenize_and_remove_stopwords


# -------------------------------------------------
# Load Model
# -------------------------------------------------

model = joblib.load(
    "models/logistic_regression.joblib"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.joblib"
)


# -------------------------------------------------
# Text Preprocessing
# -------------------------------------------------

def prepare_text(text):

    text = clean_text(text)

    text = tokenize_and_remove_stopwords(
        text
    )

    return text


# -------------------------------------------------
# Batch Prediction
# -------------------------------------------------

def predict_batch(df):

    if "text" not in df.columns:

        raise ValueError(
            "CSV must contain a 'text' column."
        )

    # Remove empty articles

    df = df.copy()

    df["text"] = df["text"].fillna("")

    df = df[
        df["text"].str.strip() != ""
    ].copy()

    if df.empty:

        raise ValueError(
            "The CSV does not contain any valid articles."
        )

    # Preprocess articles

    processed_text = df["text"].apply(
        prepare_text
    )

    # Convert to TF-IDF

    features = vectorizer.transform(
        processed_text
    )

    # Predictions

    predictions = model.predict(
        features
    )

    probabilities = model.predict_proba(
        features
    )

    # Create results

    results = pd.DataFrame({

        "Article": df["text"].values,

        "Prediction": [
            "FAKE NEWS"
            if prediction == 0
            else "REAL NEWS"
            for prediction in predictions
        ],

        "Fake Probability (%)": [
            round(probability[0] * 100, 2)
            for probability in probabilities
        ],

        "Real Probability (%)": [
            round(probability[1] * 100, 2)
            for probability in probabilities
        ],

        "Confidence (%)": [
            round(
                max(
                    probability[0],
                    probability[1]
                ) * 100,
                2
            )
            for probability in probabilities
        ]
    })

    return results