import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

def load_training_data():

    from utils.loader import load_dataset
    from utils.preprocessing import clean_text
    from utils.tokenizer import tokenize_and_remove_stopwords

    df = load_dataset()

    df["processed_text"] = df["text"].apply(
        lambda text: tokenize_and_remove_stopwords(
            clean_text(text)
        )
    )

    return df


# -------------------------------------------------
# Calculate Model Metrics
# -------------------------------------------------

def calculate_metrics():

    model = joblib.load(
        "models/logistic_regression.joblib"
    )

    vectorizer = joblib.load(
        "models/tfidf_vectorizer.joblib"
    )

    df = load_training_data()

    X = df["processed_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_test = vectorizer.transform(X_test)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions
        ),

        "Recall": recall_score(
            y_test,
            predictions
        ),

        "F1 Score": f1_score(
            y_test,
            predictions
        ),

        "AUC": roc_auc_score(
            y_test,
            probabilities
        )
    }

    return metrics