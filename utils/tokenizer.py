import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from loader import load_dataset
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
        if word.isalpha() and word.lower() not in STOP_WORDS
    ]

    return " ".join(filtered_tokens)


if __name__ == "__main__":

    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    # Load dataset using loader.py
    df = load_dataset()

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

    print("\nSample Output")
    print("-" * 60)

    print(df[["clean_text", "processed_text"]].head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nMissing Values in processed_text:")
    print(df["processed_text"].isnull().sum())

    print("\nExample Comparison")
    print("-" * 60)

    sample = df.iloc[0]

    print("\nOriginal Text:\n")
    print(sample["text"][:400])

    print("\nClean Text:\n")
    print(sample["clean_text"][:400])

    print("\nProcessed Text:\n")
    print(sample["processed_text"][:400])

    print("\nPreprocessing pipeline completed successfully.")