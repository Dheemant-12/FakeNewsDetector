import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

from loader import load_dataset
from preprocessing import clean_text
from tokenizer import tokenize_and_remove_stopwords


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

    print("Preprocessing text...")

    df["processed_text"] = df["text"].apply(prepare_text)

    print("Creating TF-IDF Vectorizer...")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    X = vectorizer.fit_transform(df["processed_text"])

    y = df["label"]

    print("\nTF-IDF Matrix Shape:")
    print(X.shape)

    print("\nNumber of Features:")
    print(len(vectorizer.get_feature_names_out()))

    print("\nFirst 20 Features:")

    print(vectorizer.get_feature_names_out()[:20])

    # Save vectorizer (small file)
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

    print("\nVectorizer saved successfully.")