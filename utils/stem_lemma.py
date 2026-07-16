import nltk

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")
nltk.download("omw-1.4")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def stem_sentence(text):

    words = text.split()

    return " ".join(
        stemmer.stem(word)
        for word in words
    )


def lemmatize_sentence(text):

    words = text.split()

    return " ".join(
        lemmatizer.lemmatize(word)
        for word in words
    )


if __name__ == "__main__":

    sample = "players were playing football studies studying cars running"

    print("=" * 50)

    print("Original:")
    print(sample)

    print("\nStemmed:")
    print(stem_sentence(sample))

    print("\nLemmatized:")
    print(lemmatize_sentence(sample))