import re
import string
import pandas as pd


def clean_text(text):
    """
    Cleans a news article for machine learning.
    """

    # Convert to string
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove URLs
    text = re.sub(r"http\\S+|www\\S+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":

    fake_df = pd.read_csv("data/Fake.csv")
    true_df = pd.read_csv("data/True.csv")

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)

    print("Cleaning text...")

    df["clean_text"] = df["text"].apply(clean_text)

    print("\nSample Cleaned Text")

    print("-" * 60)

    print(df[["text", "clean_text"]].head())

    print("\nCleaning completed successfully.")