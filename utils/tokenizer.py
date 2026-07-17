import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from utils.loader import load_dataset
from utils.preprocessing import clean_text

# Download NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

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

    df = load_dataset()

    print(f"Total Articles: {len(df)}")

    print("\nCleaning text...")

    df["clean_text"] = df["text"].apply(clean_text)

    print("Cleaning completed.")

    print("\nTokenizing and removing stopwords...")

    df["processed_text"] = df["clean_text"].apply(
        tokenize_and_remove_stopwords
    )

    print("Tokenization completed.")

    print("\nSample Output")
    print("-" * 60)

    print(df[["clean_text", "processed_text"]].head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nMissing Values:")
    print(df["processed_text"].isnull().sum())

    print("\nPreprocessing pipeline completed successfully.")