import os
import sys
import json

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


# =================================================
# Add Project Root
# =================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =================================================
# Project Imports
# =================================================

from utils.loader import load_dataset
from utils.preprocessing import clean_text
from utils.tokenizer import tokenize_and_remove_stopwords


# =================================================
# Text Preprocessing
# =================================================

def prepare_text(text):
    """
    Complete text preprocessing pipeline.
    """

    text = clean_text(text)

    text = tokenize_and_remove_stopwords(
        text
    )

    return text


# =================================================
# Load Dataset
# =================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = load_dataset()

print(
    f"Dataset Loaded Successfully!"
)

print(
    f"Total Articles : {len(df)}"
)


# =================================================
# Preprocess Dataset
# =================================================

print("\nCleaning and preprocessing text...")

df["processed_text"] = df["text"].apply(
    prepare_text
)

X = df["processed_text"]

y = df["label"]


# =================================================
# Train/Test Split
# =================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(
    f"Training Samples : {len(X_train)}"
)

print(
    f"Testing Samples  : {len(X_test)}"
)


# =================================================
# TF-IDF
# =================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
)

X_train_vectorized = vectorizer.fit_transform(
    X_train
)

X_test_vectorized = vectorizer.transform(
    X_test
)

print(
    f"Feature Count : {X_train_vectorized.shape[1]}"
)


# =================================================
# Models
# =================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),

    "Naive Bayes":
        MultinomialNB(),

    "Linear SVM":
        LinearSVC(
            random_state=42,
        ),
}


# =================================================
# Model Comparison
# =================================================

results = []

print("\n")
print("=" * 60)
print("TRAINING MODELS")
print("=" * 60)


for name, model in models.items():

    print(
        f"\nTraining {name}..."
    )

    # ---------------------------------------------
    # Train
    # ---------------------------------------------

    model.fit(
        X_train_vectorized,
        y_train
    )

    # ---------------------------------------------
    # Predictions
    # ---------------------------------------------

    predictions = model.predict(
        X_test_vectorized
    )

    # ---------------------------------------------
    # Accuracy
    # ---------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # ---------------------------------------------
    # Precision
    # ---------------------------------------------

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    # ---------------------------------------------
    # Recall
    # ---------------------------------------------

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    # ---------------------------------------------
    # F1
    # ---------------------------------------------

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    # ---------------------------------------------
    # AUC
    # ---------------------------------------------

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            X_test_vectorized
        )[:, 1]

    else:

        probabilities = model.decision_function(
            X_test_vectorized
        )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    # ---------------------------------------------
    # Store Results
    # ---------------------------------------------

    results.append({

        "Model": name,

        "Accuracy": round(
            accuracy,
            4
        ),

        "Precision": round(
            precision,
            4
        ),

        "Recall": round(
            recall,
            4
        ),

        "F1 Score": round(
            f1,
            4
        ),

        "AUC": round(
            auc,
            4
        ),
    })

    print(
        f"{name} completed."
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"AUC       : {auc:.4f}"
    )


# =================================================
# Create Results DataFrame
# =================================================

results_df = pd.DataFrame(
    results
)


# =================================================
# Sort by F1 Score
# =================================================

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(
    drop=True
)


# =================================================
# Display Results
# =================================================

print("\n")
print("=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# =================================================
# Best Model
# =================================================

best_model = results_df.iloc[0]

print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    f"Model : {best_model['Model']}"
)

print(
    f"Accuracy : {best_model['Accuracy']:.4f}"
)

print(
    f"Precision : {best_model['Precision']:.4f}"
)

print(
    f"Recall : {best_model['Recall']:.4f}"
)

print(
    f"F1 Score : {best_model['F1 Score']:.4f}"
)

print(
    f"AUC : {best_model['AUC']:.4f}"
)


# =================================================
# Save Results
# =================================================

results_directory = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(
    results_directory,
    exist_ok=True
)


# -------------------------------------------------
# Save CSV
# -------------------------------------------------

csv_path = os.path.join(
    results_directory,
    "model_comparison.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)


# -------------------------------------------------
# Save JSON
# -------------------------------------------------

json_path = os.path.join(
    results_directory,
    "model_comparison.json"
)

comparison_data = {

    "best_model":
        best_model["Model"],

    "models":
        results
}


with open(
    json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        comparison_data,
        file,
        indent=4
    )


# =================================================
# Final Message
# =================================================

print("\n")
print("=" * 60)
print("MODEL COMPARISON COMPLETED")
print("=" * 60)

print(
    f"CSV saved to : {csv_path}"
)

print(
    f"JSON saved to : {json_path}"
)