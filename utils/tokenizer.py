import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Import the cleaning function from Day 3
from preprocessing import clean_text

# Download NLTK resources (only required the first time)
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Load English stopwords
STOP_WORDS = set(stopwords.words("english"))


def tokenize_and_remove_stopwords(text):
    """
    Tokenize text and remove English stopwords.
    """

    tokens = word_tokenize(str(text))

    filtered_tokens = [
        word
        for word in tokens
        if word.isalpha() and word not in STOP_WORDS
    ]

    return " ".join(filtered_tokens)


if __name__ == "__main__":

    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    # Load original datasets
    fake_df = pd.read_csv("data/Fake.csv")
    true_df = pd.read_csv("data/True.csv")

    # Add labels
    fake_df["label"] = 0
    true_df["label"] = 1

    # Combine datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)

    print(f"Total Articles: {len(df)}")

    print("\nCleaning text...")

    # Apply Day 3 preprocessing
    df["clean_text"] = df["text"].apply(clean_text)

    print("Cleaning completed.")

    print("\nTokenizing and removing stopwords...")

    # Apply Day 4 preprocessing
    df["processed_text"] = df["clean_text"].apply(
        tokenize_and_remove_stopwords
    )

    print("Tokenization completed.")

    print("\nSample Output:")
    print("-" * 60)

    print(df[["text", "clean_text", "processed_text"]].head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nMissing Values:")
    print(df["processed_text"].isnull().sum())

    print("\nPreprocessing pipeline completed successfully.")