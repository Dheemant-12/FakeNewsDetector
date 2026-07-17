import os
import sys
import joblib

# ---------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------
# Imports
# ---------------------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from utils.loader import load_dataset
from utils.preprocessing import clean_text
from utils.tokenizer import tokenize_and_remove_stopwords


def prepare_text(text):
    """
    Complete preprocessing pipeline.
    """
    text = clean_text(text)
    text = tokenize_and_remove_stopwords(text)
    return text


if __name__ == "__main__":

    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    df = load_dataset()

    print("Dataset Loaded Successfully!")
    print(f"Total Articles : {len(df)}")

    print("\nCleaning and preprocessing text...")

    df["processed_text"] = df["text"].apply(prepare_text)

    X = df["processed_text"]
    y = df["label"]

    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    print("\nCreating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, predictions))

    # Create models folder if it doesn't exist
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/logistic_regression.joblib")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

    print("\nModel saved successfully!")
    print("Vectorizer saved successfully!")