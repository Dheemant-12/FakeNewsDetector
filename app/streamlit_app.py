import os
import sys
import re
import joblib
import streamlit as st
import pandas as pd

# -------------------------------------------------
# Add Project Root
# -------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.preprocessing import clean_text
from utils.tokenizer import tokenize_and_remove_stopwords


# -------------------------------------------------
# Text Preprocessing
# -------------------------------------------------

def prepare_text(text):
    """
    Complete preprocessing pipeline.
    """
    text = clean_text(text)
    text = tokenize_and_remove_stopwords(text)
    return text


# -------------------------------------------------
# Input Validation
# -------------------------------------------------

def validate_input(text):
    """
    Validate user input before prediction.
    Returns (is_valid, message)
    """

    text = text.strip()

    if not text:
        return False, "Please enter a news article."

    if not re.search(r"[A-Za-z]", text):
        return False, "Input should contain alphabetic text."

    if len(text.split()) < 8:
        return False, "Please enter a longer news article (at least 8 words)."

    return True, ""


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
# Session State
# -------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detector")

st.write(
    "Paste a news article below and the model will predict whether it is Fake or Real."
)

# -------------------------------------------------
# Day 18 - Model Information
# -------------------------------------------------

st.divider()

st.subheader("📊 Model Information")

col1, col2 = st.columns(2)

with col1:
    st.metric("Model", "Logistic Regression")
    st.metric("Vectorizer", "TF-IDF")

with col2:
    st.metric("Features", "5000")
    st.metric("Classes", "Fake / Real")

st.info(
    "This application uses a TF-IDF Vectorizer to convert news articles into numerical "
    "features and a Logistic Regression model to classify them as Fake or Real."
)

with st.expander("ℹ️ How does the model work?"):

    st.markdown("""
### Prediction Pipeline

1. User enters a news article.
2. Text is cleaned and normalized.
3. Stop words are removed.
4. TF-IDF converts the article into numerical features.
5. Logistic Regression predicts Fake or Real news.
6. Confidence scores are displayed.
7. Users can download a prediction report.
""")

st.divider()


# -------------------------------------------------
# User Input
# -------------------------------------------------

article = st.text_area(
    "Enter News Article",
    height=250
)


# -------------------------------------------------
# Prediction
# -------------------------------------------------

if st.button("Predict"):

    is_valid, message = validate_input(article)

    if not is_valid:
        st.warning(message)

    else:

        try:

            # -----------------------------
            # Preprocess
            # -----------------------------

            processed = prepare_text(article)

            # -----------------------------
            # Vectorize
            # -----------------------------

            features = vectorizer.transform([processed])

            # -----------------------------
            # Predict
            # -----------------------------

            prediction = model.predict(features)[0]

            probabilities = model.predict_proba(features)[0]

            fake_probability = probabilities[0] * 100
            real_probability = probabilities[1] * 100

            # -----------------------------
            # Display Prediction
            # -----------------------------

            if prediction == 0:
                st.error("🚨 Prediction: FAKE NEWS")
            else:
                st.success("✅ Prediction: REAL NEWS")

            # -----------------------------
            # Confidence Scores
            # -----------------------------

            st.subheader("Prediction Confidence")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Fake", f"{fake_probability:.2f}%")
                st.progress(min(int(fake_probability), 100))

            with col2:
                st.metric("Real", f"{real_probability:.2f}%")
                st.progress(min(int(real_probability), 100))

            st.info(
                "The model assigns probabilities to both classes. "
                "The class with the higher probability becomes the final prediction."
            )
            # -----------------------------
            # Download Report
            # -----------------------------

            report = pd.DataFrame({
                "Article": [article],
                "Prediction": [
                    "FAKE NEWS" if prediction == 0 else "REAL NEWS"
                ],
                "Fake Probability (%)": [
                    round(fake_probability, 2)
                ],
                "Real Probability (%)": [
                    round(real_probability, 2)
                ]
            })

            csv = report.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Prediction Report",
                data=csv,
                file_name="prediction_report.csv",
                mime="text/csv"
            )

            # -----------------------------
            # Save Prediction History
            # -----------------------------

            st.session_state.history.append({
                "Prediction": "FAKE NEWS" if prediction == 0 else "REAL NEWS",
                "Fake Probability (%)": round(fake_probability, 2),
                "Real Probability (%)": round(real_probability, 2),
                "Article": article[:100] + "..." if len(article) > 100 else article
            })

            # -----------------------------
            # Debug Information
            # -----------------------------

            with st.expander("🛠 Debug Information"):

                st.write("**Processed Text:**")
                st.code(processed)

                st.write(f"**Feature Shape:** {features.shape}")
                st.write(f"**Non-zero Features:** {features.nnz}")

        except Exception as e:
            st.error("An unexpected error occurred while making the prediction.")
            st.exception(e)

# -------------------------------------------------
# Prediction History
# -------------------------------------------------

if st.session_state.history:

    st.divider()
    st.subheader("📜 Prediction History")

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button("🗑 Clear History"):
        st.session_state.history = []
        st.rerun()