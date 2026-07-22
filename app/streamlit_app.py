import os
import sys
import joblib
import streamlit as st

# -------------------------------------------------
# Add Project Root
# -------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.preprocessing import clean_text
from utils.tokenizer import tokenize_and_remove_stopwords


def prepare_text(text):
    text = clean_text(text)
    text = tokenize_and_remove_stopwords(text)
    return text


# -------------------------------------------------
# Load Model Once
# -------------------------------------------------

@st.cache_resource
def load_models():
    model = joblib.load("models/logistic_regression.joblib")
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    return model, vectorizer


model, vectorizer = load_models()

# -------------------------------------------------
# UI
# -------------------------------------------------

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detector")

st.write(
    "Paste a news article below and the model will predict "
    "whether it is Fake or Real."
)

article = st.text_area(
    "Enter News Article",
    height=250
)

if st.button("Predict"):

    if article.strip() == "":
        st.warning("Please enter some text.")
    else:

        processed = prepare_text(article)

        features = vectorizer.transform([processed])

        prediction = model.predict(features)[0]

        confidence = model.predict_proba(features).max() * 100

        if prediction == 0:
            st.error("🚨 Prediction: FAKE NEWS")
        else:
            st.success("✅ Prediction: REAL NEWS")

        st.info(f"Confidence: {confidence:.2f}%")