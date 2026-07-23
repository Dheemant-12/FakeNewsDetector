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

        # -------------------------------
        # Preprocess
        # -------------------------------

        processed = prepare_text(article)

        # -------------------------------
        # Vectorize
        # -------------------------------

        features = vectorizer.transform([processed])

        # -------------------------------
        # Prediction
        # -------------------------------

        prediction = model.predict(features)[0]

        probabilities = model.predict_proba(features)[0]

        fake_probability = probabilities[0] * 100
        real_probability = probabilities[1] * 100

        # -------------------------------
        # Display Result
        # -------------------------------

        if prediction == 0:
            st.error("🚨 Prediction: FAKE NEWS")
        else:
            st.success("✅ Prediction: REAL NEWS")

        st.subheader("Prediction Probabilities")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Fake", f"{fake_probability:.2f}%")

        with col2:
            st.metric("Real", f"{real_probability:.2f}%")

        st.divider()

        st.subheader("Debug Information")

        st.write("**Processed Text:**")
        st.code(processed)

        st.write(f"**Feature Shape:** {features.shape}")
        st.write(f"**Non-zero Features:** {features.nnz}")