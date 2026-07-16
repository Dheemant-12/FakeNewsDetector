import re
import string


def clean_text(text):
    """
    Cleans a news article for machine learning.
    """

    text = str(text)

    text = text.lower()

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\s+", " ", text).strip()

    return text