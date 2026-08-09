import os
import sys
import re
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------------------------------
# Add Project Root FIRST
# -------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------
# Project Imports
# -------------------------------------------------

from model_metrics import load_metrics
from explainability.lime_explainer import explain_prediction
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
# Load Model
# -------------------------------------------------

@st.cache_resource
def load_models():

    model = joblib.load(
        "models/logistic_regression.joblib"
    )

    vectorizer = joblib.load(
        "models/tfidf_vectorizer.joblib"
    )

    return model, vectorizer


model, vectorizer = load_models()


# -------------------------------------------------
# Session State
# -------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)


# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.title("📰 Fake News Detector")

st.sidebar.markdown("---")

st.sidebar.subheader("👨‍💻 Developer")
st.sidebar.write("Dheemant Reddy")

st.sidebar.subheader("🤖 Model")
st.sidebar.write("Logistic Regression")

st.sidebar.subheader("📚 Vectorizer")
st.sidebar.write("TF-IDF")

st.sidebar.subheader("🛠 Tech Stack")

st.sidebar.markdown("""
- Python
- Streamlit
- Scikit-learn
- Pandas
- Joblib
- LIME
- Matplotlib
""")

st.sidebar.subheader("📌 Version")
st.sidebar.write("v1.0")

st.sidebar.markdown("---")

st.sidebar.info(
    "This project predicts whether a news article is Fake or Real "
    "using Machine Learning."
)

st.sidebar.markdown("---")

st.sidebar.caption("Made with ❤️ using Streamlit")


# -------------------------------------------------
# Main UI
# -------------------------------------------------

st.title("📰 Fake News Detector")

st.write(
    "Paste a news article below and the model will predict "
    "whether it is Fake or Real."
)

st.divider()


# -------------------------------------------------
# Model Information
# -------------------------------------------------

st.subheader("📊 Model Information")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Model",
        "Logistic Regression"
    )

    st.metric(
        "Vectorizer",
        "TF-IDF"
    )


with col2:

    st.metric(
        "Features",
        "5000"
    )

    st.metric(
        "Classes",
        "Fake / Real"
    )


st.info(
    "This application uses a TF-IDF Vectorizer to convert text "
    "into numerical features and a Logistic Regression model "
    "to classify news articles."
)


with st.expander("ℹ️ How does the model work?"):

    st.markdown("""
### Prediction Pipeline

1. User enters a news article.
2. Text is cleaned.
3. Stop words are removed.
4. TF-IDF converts text into numerical vectors.
5. Logistic Regression predicts Fake or Real.
6. Confidence scores are displayed.
7. LIME explains the prediction.
8. Prediction reports and history can be downloaded.
""")


st.divider()
# -------------------------------------------------
# Model Performance Dashboard
# -------------------------------------------------

st.subheader("📈 Model Performance")

try:

    metrics = load_metrics()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Accuracy",
            f"{metrics['accuracy'] * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Precision",
            f"{metrics['precision'] * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Recall",
            f"{metrics['recall'] * 100:.2f}%"
        )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "F1 Score",
            f"{metrics['f1_score'] * 100:.2f}%"
        )

    with col2:

        st.metric(
            "AUC",
            f"{metrics['auc']:.4f}"
        )

    st.info(
        "These metrics were calculated during model evaluation "
        "and saved for fast access."
    )

except Exception:

    st.warning(
        "Model performance metrics could not be loaded."
    )


# -------------------------------------------------
# User Input
# -------------------------------------------------

st.divider()

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

            # -------------------------------------------------
            # Preprocess
            # -------------------------------------------------

            processed = prepare_text(article)

            # -------------------------------------------------
            # Vectorize
            # -------------------------------------------------

            features = vectorizer.transform(
                [processed]
            )

            # -------------------------------------------------
            # Prediction
            # -------------------------------------------------

            prediction = model.predict(
                features
            )[0]

            probabilities = model.predict_proba(
                features
            )[0]

            fake_probability = (
                probabilities[0] * 100
            )

            real_probability = (
                probabilities[1] * 100
            )

            confidence = max(
                fake_probability,
                real_probability
            )


            # -------------------------------------------------
            # Confidence Level
            # -------------------------------------------------

            if confidence >= 90:

                confidence_level = "🟢 High"

            elif confidence >= 70:

                confidence_level = "🟡 Medium"

            else:

                confidence_level = "🔴 Low"


            # -------------------------------------------------
            # Display Prediction
            # -------------------------------------------------

            if prediction == 0:

                st.error(
                    "🚨 Prediction: FAKE NEWS"
                )

            else:

                st.success(
                    "✅ Prediction: REAL NEWS"
                )


            # -------------------------------------------------
            # Prediction Confidence
            # -------------------------------------------------

            st.subheader(
                "Prediction Confidence"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Fake",
                    f"{fake_probability:.2f}%"
                )

                st.progress(
                    min(
                        int(fake_probability),
                        100
                    )
                )

            with col2:

                st.metric(
                    "Real",
                    f"{real_probability:.2f}%"
                )

                st.progress(
                    min(
                        int(real_probability),
                        100
                    )
                )

            st.info(
                "The model assigns probabilities to both classes. "
                "The class with the higher probability becomes "
                "the final prediction."
            )


            # -------------------------------------------------
            # Confidence Level
            # -------------------------------------------------

            st.subheader(
                "Confidence Level"
            )

            st.success(
                f"{confidence_level} "
                f"Confidence ({confidence:.2f}%)"
            )


            # -------------------------------------------------
            # Overall Confidence Gauge
            # -------------------------------------------------

            st.subheader(
                "🎯 Overall Prediction Confidence"
            )

            st.progress(
                min(
                    int(confidence),
                    100
                )
            )

            st.metric(
                "Confidence Score",
                f"{confidence:.2f}%"
            )

            if confidence >= 95:

                st.success(
                    "The model is extremely confident "
                    "about this prediction."
                )

            elif confidence >= 80:

                st.info(
                    "The model is reasonably confident "
                    "about this prediction."
                )

            else:

                st.warning(
                    "This prediction has relatively low "
                    "confidence. Interpret the result carefully."
                )


            # -------------------------------------------------
            # Prediction Report
            # -------------------------------------------------

            report = pd.DataFrame({

                "Article": [article],

                "Prediction": [
                    "FAKE NEWS"
                    if prediction == 0
                    else "REAL NEWS"
                ],

                "Fake Probability (%)": [
                    round(
                        fake_probability,
                        2
                    )
                ],

                "Real Probability (%)": [
                    round(
                        real_probability,
                        2
                    )
                ]
            })

            csv = report.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Download Prediction Report",
                data=csv,
                file_name="prediction_report.csv",
                mime="text/csv"
            )


            # -------------------------------------------------
            # LIME Explanation
            # -------------------------------------------------

            st.subheader(
                "🧠 Why did the model predict this?"
            )

            explanation = explain_prediction(
                processed
            )

            lime_df = pd.DataFrame(
                explanation.as_list(),
                columns=[
                    "Word",
                    "Importance"
                ]
            )

            st.dataframe(
                lime_df,
                use_container_width=True,
                hide_index=True
            )


            # -------------------------------------------------
            # LIME Graph
            # -------------------------------------------------

            fig, ax = plt.subplots(
                figsize=(8, 4)
            )

            lime_df = lime_df.sort_values(
                "Importance"
            )

            ax.barh(
                lime_df["Word"],
                lime_df["Importance"]
            )

            ax.set_title(
                "Top Words Influencing Prediction"
            )

            ax.set_xlabel(
                "Importance Score"
            )

            ax.axvline(
                0,
                color="black",
                linewidth=1
            )

            st.pyplot(fig)

            plt.close(fig)
                        # -------------------------------------------------
            # Save Prediction History
            # -------------------------------------------------

            st.session_state.history.append({

                "Time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                "Prediction": (
                    "FAKE NEWS"
                    if prediction == 0
                    else "REAL NEWS"
                ),

                "Confidence (%)": round(
                    confidence,
                    2
                ),

                "Fake Probability (%)": round(
                    fake_probability,
                    2
                ),

                "Real Probability (%)": round(
                    real_probability,
                    2
                ),

                "Article": (
                    article[:100] + "..."
                    if len(article) > 100
                    else article
                )
            })


            # -------------------------------------------------
            # Export Session History
            # -------------------------------------------------

            history_df = pd.DataFrame(
                st.session_state.history
            )

            st.subheader(
                "📥 Export Session History"
            )

            history_csv = history_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Download Session History",
                data=history_csv,
                file_name="prediction_history.csv",
                mime="text/csv"
            )


            # -------------------------------------------------
            # Important Words
            # -------------------------------------------------

            st.subheader(
                "🔍 Important Words"
            )

            important_words = processed.split()[:15]

            if important_words:

                st.write(
                    ", ".join(important_words)
                )

            else:

                st.write(
                    "No important words found."
                )


            # -------------------------------------------------
            # Debug Information
            # -------------------------------------------------

            with st.expander(
                "🛠 Debug Information"
            ):

                st.write(
                    "**Processed Text:**"
                )

                st.code(processed)

                st.write(
                    f"**Feature Shape:** {features.shape}"
                )

                st.write(
                    f"**Non-zero Features:** {features.nnz}"
                )


        except Exception as e:

            st.error(
                "An unexpected error occurred "
                "while making the prediction."
            )

            st.exception(e)


# =================================================
# Session Statistics
# =================================================

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    total_predictions = len(
        history_df
    )

    fake_predictions = (
        history_df["Prediction"] == "FAKE NEWS"
    ).sum()

    real_predictions = (
        history_df["Prediction"] == "REAL NEWS"
    ).sum()


    # -------------------------------------------------
    # Percentages
    # -------------------------------------------------

    fake_percent = (
        fake_predictions /
        total_predictions
    ) * 100

    real_percent = (
        real_predictions /
        total_predictions
    ) * 100


    # -------------------------------------------------
    # Statistics
    # -------------------------------------------------

    st.divider()

    st.subheader(
        "📈 Session Statistics"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total",
            total_predictions
        )

    with col2:

        st.metric(
            "Fake",
            fake_predictions
        )

    with col3:

        st.metric(
            "Real",
            real_predictions
        )


    # -------------------------------------------------
    # Fake Percentage
    # -------------------------------------------------

    st.write(
        "**Fake Predictions**"
    )

    st.progress(
        min(
            int(fake_percent),
            100
        )
    )

    st.caption(
        f"{fake_percent:.1f}%"
    )


    # -------------------------------------------------
    # Real Percentage
    # -------------------------------------------------

    st.write(
        "**Real Predictions**"
    )

    st.progress(
        min(
            int(real_percent),
            100
        )
    )

    st.caption(
        f"{real_percent:.1f}%"
    )


    st.divider()


    # -------------------------------------------------
    # Prediction History
    # -------------------------------------------------

    st.subheader(
        "📜 Prediction History"
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )


    # -------------------------------------------------
    # Download Full History
    # -------------------------------------------------

    full_history_csv = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Full Prediction History",
        data=full_history_csv,
        file_name="full_prediction_history.csv",
        mime="text/csv"
    )


    # -------------------------------------------------
    # Clear History
    # -------------------------------------------------

    if st.button(
        "🗑 Clear History"
    ):

        st.session_state.history = []

        st.rerun()