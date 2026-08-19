import os
import sys
import re
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_metrics import load_metrics
from explainability.lime_explainer import explain_prediction
from utils.preprocessing import clean_text
from utils.tokenizer import tokenize_and_remove_stopwords
from batch_predict import predict_batch

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")


def prepare_text(text):
    text = clean_text(text)
    return tokenize_and_remove_stopwords(text)


def validate_input(text):
    text = text.strip()
    if not text:
        return False, "Please enter a news article."
    if not re.search(r"[A-Za-z]", text):
        return False, "Input should contain alphabetic text."
    if len(text.split()) < 8:
        return False, "Please enter a longer news article (at least 8 words)."
    return True, ""


@st.cache_resource
def load_models():
    model = joblib.load("models/logistic_regression.joblib")
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    return model, vectorizer


model, vectorizer = load_models()


@st.cache_data
def get_model_metrics():
    return load_metrics()


try:
    metrics = get_model_metrics()
except Exception:
    metrics = None


if "history" not in st.session_state:
    st.session_state.history = []


# =================================================
# Sidebar
# =================================================

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
st.sidebar.info("This project predicts whether a news article is Fake or Real using Machine Learning.")
st.sidebar.markdown("---")
st.sidebar.caption("Made with ❤️ using Streamlit")


# =================================================
# Header + Navigation
# =================================================

st.title("📰 Fake News Detector")
st.write("Machine Learning powered Fake News Detection and Analysis System.")

tab_dashboard, tab_single, tab_batch = st.tabs([
    "📊 Dashboard",
    "🔎 Single Prediction",
    "📂 Batch Prediction"
])


# =================================================
# DASHBOARD TAB
# =================================================

with tab_dashboard:

    st.header("📊 Model Dashboard")
    st.write("Overview of the Fake News Detection model and its performance.")
    st.divider()

    st.subheader("🤖 Model Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model", "Logistic Regression")
    with col2:
        st.metric("Vectorizer", "TF-IDF")
    with col3:
        st.metric("Features", "5000")
    with col4:
        st.metric("Classes", "Fake / Real")

    st.info(
        "The application uses TF-IDF to convert news text into numerical features "
        "and Logistic Regression to classify the article."
    )

    with st.expander("ℹ️ How does the model work?"):
        st.markdown("""
### Prediction Pipeline

1. User enters a news article.
2. Text is cleaned and normalized.
3. Stop words are removed.
4. TF-IDF converts the text into numerical features.
5. Logistic Regression predicts Fake or Real.
6. Confidence scores are displayed.
7. LIME explains the prediction.
8. Prediction history is stored for the session.
9. CSV files can be analyzed in batch mode.
""")

    st.divider()
    st.subheader("📈 Model Performance")
    # =================================================
    # Classification Summary
    # =================================================

    st.divider()

    st.subheader(
        "📋 Classification Summary"
    )

    if metrics is not None:

        summary_data = {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "AUC"
            ],
            "Score": [
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
                metrics["auc"]
            ]
        }

        summary_df = pd.DataFrame(
            summary_data
        )

        summary_df["Score"] = summary_df[
            "Score"
        ].round(4)

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Classification metrics are not available."
        )
    # =================================================
    # Performance Interpretation
    # =================================================

    if metrics is not None:

        accuracy = metrics["accuracy"]

        if accuracy >= 0.95:

            st.success(
                f"🟢 Excellent model performance: "
                f"{accuracy * 100:.2f}% accuracy."
            )

        elif accuracy >= 0.85:

            st.info(
                f"🟡 Good model performance: "
                f"{accuracy * 100:.2f}% accuracy."
            )

        else:

            st.warning(
                f"🔴 The model achieved "
                f"{accuracy * 100:.2f}% accuracy. "
                "Further model improvement may be required."
            )

    if metrics is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        with col2:
            st.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
        with col3:
            st.metric("Recall", f"{metrics['recall'] * 100:.2f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("F1 Score", f"{metrics['f1_score'] * 100:.2f}%")
        with col2:
            st.metric("AUC", f"{metrics['auc']:.4f}")
    else:
        st.warning("Model performance metrics could not be loaded.")

    if metrics is not None:
        st.divider()
        st.subheader("🔲 Confusion Matrix")

        matrix = metrics.get("confusion_matrix")

        if matrix:
            confusion_df = pd.DataFrame(
                matrix,
                index=["Actual Fake", "Actual Real"],
                columns=["Predicted Fake", "Predicted Real"]
            )
            st.dataframe(confusion_df, use_container_width=True)
        else:
            st.info("Confusion matrix data is not available.")

        st.divider()
        st.subheader("📈 ROC Curve")

        roc_data = metrics.get("roc_curve")

        if roc_data:
            fpr = roc_data.get("fpr", [])
            tpr = roc_data.get("tpr", [])

            if fpr and tpr:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.4f}")
                ax.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("ROC curve data is empty.")
        else:
            st.info("ROC curve data is not available.")


# =================================================
# SINGLE PREDICTION TAB
# =================================================

with tab_single:

    st.header("🔎 Single Article Prediction")
    st.write("Enter one news article and let the model analyze it.")
    st.divider()

    article = st.text_area(
        "Enter News Article",
        height=250,
        placeholder="Paste a news article here..."
    )

    if st.button("🚀 Predict", key="single_predict"):

        is_valid, message = validate_input(article)

        if not is_valid:
            st.warning(message)

        else:
            try:
                processed = prepare_text(article)
                features = vectorizer.transform([processed])
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]

                fake_probability = probabilities[0] * 100
                real_probability = probabilities[1] * 100
                confidence = max(fake_probability, real_probability)

                if confidence >= 90:
                    confidence_level = "🟢 High"
                elif confidence >= 70:
                    confidence_level = "🟡 Medium"
                else:
                    confidence_level = "🔴 Low"

                st.subheader("Prediction Result")

                if prediction == 0:
                    st.error("🚨 Prediction: FAKE NEWS")
                else:
                    st.success("✅ Prediction: REAL NEWS")

                st.subheader("Prediction Confidence")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Fake", f"{fake_probability:.2f}%")
                    st.progress(min(int(fake_probability), 100))

                with col2:
                    st.metric("Real", f"{real_probability:.2f}%")
                    st.progress(min(int(real_probability), 100))

                st.subheader("🎯 Confidence Level")

                if confidence_level == "🟢 High":
                    st.success(f"{confidence_level} Confidence ({confidence:.2f}%)")
                elif confidence_level == "🟡 Medium":
                    st.warning(f"{confidence_level} Confidence ({confidence:.2f}%)")
                else:
                    st.error(f"{confidence_level} Confidence ({confidence:.2f}%)")

                st.info("The class with the higher probability becomes the final prediction.")

                report = pd.DataFrame({
                    "Article": [article],
                    "Prediction": ["FAKE NEWS" if prediction == 0 else "REAL NEWS"],
                    "Fake Probability (%)": [round(fake_probability, 2)],
                    "Real Probability (%)": [round(real_probability, 2)],
                    "Confidence (%)": [round(confidence, 2)]
                })

                st.download_button(
                    "📥 Download Prediction Report",
                    report.to_csv(index=False).encode("utf-8"),
                    "prediction_report.csv",
                    "text/csv",
                    key="single_download"
                )

                st.divider()
                st.subheader("🧠 Why did the model predict this?")

                explanation = explain_prediction(processed)

                lime_df = pd.DataFrame(
                    explanation.as_list(),
                    columns=["Word", "Importance"]
                )

                st.dataframe(
                    lime_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader("📊 Word Importance")

                fig, ax = plt.subplots(figsize=(8, 4))
                lime_df = lime_df.sort_values("Importance")
                ax.barh(lime_df["Word"], lime_df["Importance"])
                ax.axvline(0, linewidth=1)
                ax.set_xlabel("Importance Score")
                ax.set_title("Top Words Influencing Prediction")
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("🔍 Important Words")

                important_words = processed.split()[:15]

                st.write(
                    ", ".join(important_words)
                    if important_words
                    else "No important words found."
                )

                st.session_state.history.append({
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Prediction": "FAKE NEWS" if prediction == 0 else "REAL NEWS",
                    "Confidence (%)": round(confidence, 2),
                    "Fake Probability (%)": round(fake_probability, 2),
                    "Real Probability (%)": round(real_probability, 2),
                    "Article": article[:100] + "..." if len(article) > 100 else article
                })

                with st.expander("🛠 Debug Information"):
                    st.write("**Processed Text:**")
                    st.code(processed)
                    st.write(f"**Feature Shape:** {features.shape}")
                    st.write(f"**Non-zero Features:** {features.nnz}")

            except Exception as e:
                st.error("An unexpected error occurred while making the prediction.")
                st.exception(e)

    if st.session_state.history:

        st.divider()
        st.subheader("📜 Prediction History")

        history_df = pd.DataFrame(st.session_state.history)

        total_predictions = len(history_df)
        fake_predictions = (history_df["Prediction"] == "FAKE NEWS").sum()
        real_predictions = (history_df["Prediction"] == "REAL NEWS").sum()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total", total_predictions)
        with col2:
            st.metric("Fake", fake_predictions)
        with col3:
            st.metric("Real", real_predictions)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Download Prediction History",
            history_df.to_csv(index=False).encode("utf-8"),
            "prediction_history.csv",
            "text/csv",
            key="history_download"
        )

        if st.button("🗑 Clear History", key="clear_history"):
            st.session_state.history = []
            st.rerun()


# =================================================
# BATCH PREDICTION TAB
# =================================================

with tab_batch:

    st.header("📂 Batch News Prediction")

    st.write(
        "Upload a CSV containing a column named 'text' "
        "to predict multiple news articles."
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="batch_csv"
    )

    if uploaded_file is not None:

        try:
            batch_df = pd.read_csv(uploaded_file)

            if "text" not in batch_df.columns:

                st.error("❌ CSV must contain a 'text' column.")

            else:

                st.write(f"📄 Articles found: {len(batch_df)}")

                with st.expander("👀 Preview Uploaded Data"):
                    st.dataframe(
                        batch_df.head(10),
                        use_container_width=True,
                        hide_index=True
                    )

                if st.button("🚀 Predict Batch", key="batch_predict"):

                    results = predict_batch(batch_df)

                    if results.empty:

                        st.warning(
                            "No valid articles were found in the uploaded CSV."
                        )

                    else:

                        st.success(
                            f"Successfully predicted {len(results)} articles."
                        )

                        st.subheader("📊 Batch Analytics Dashboard")

                        fake_count = (
                            results["Prediction"] == "FAKE NEWS"
                        ).sum()

                        real_count = (
                            results["Prediction"] == "REAL NEWS"
                        ).sum()

                        total_count = len(results)

                        average_confidence = (
                            results["Confidence (%)"].mean()
                        )

                        fake_percentage = (
                            fake_count / total_count * 100
                        )

                        real_percentage = (
                            real_count / total_count * 100
                        )

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Total Articles", total_count)

                        with col2:
                            st.metric(
                                "Fake News",
                                fake_count,
                                f"{fake_percentage:.1f}%"
                            )

                        with col3:
                            st.metric(
                                "Real News",
                                real_count,
                                f"{real_percentage:.1f}%"
                            )

                        with col4:
                            st.metric(
                                "Avg Confidence",
                                f"{average_confidence:.1f}%"
                            )

                        st.subheader("📈 Prediction Distribution")

                        prediction_chart = pd.DataFrame(
                            {"Articles": [fake_count, real_count]},
                            index=["FAKE NEWS", "REAL NEWS"]
                        )

                        st.bar_chart(prediction_chart)

                        st.subheader("🎯 Confidence Distribution")

                        confidence_chart = pd.DataFrame({
                            "Confidence (%)":
                                results["Confidence (%)"].values
                        })

                        st.line_chart(confidence_chart)

                        high_confidence = (
                            results["Confidence (%)"] >= 90
                        ).sum()

                        medium_confidence = (
                            (results["Confidence (%)"] >= 70)
                            &
                            (results["Confidence (%)"] < 90)
                        ).sum()

                        low_confidence = (
                            results["Confidence (%)"] < 70
                        ).sum()

                        st.subheader("🎯 Confidence Categories")

                        confidence_categories = pd.DataFrame(
                            {"Articles": [
                                high_confidence,
                                medium_confidence,
                                low_confidence
                            ]},
                            index=[
                                "High (90%+)",
                                "Medium (70-89%)",
                                "Low (<70%)"
                            ]
                        )

                        st.bar_chart(confidence_categories)

                        if low_confidence > 0:
                            st.warning(
                                f"{low_confidence} article(s) have low prediction "
                                "confidence. Review them carefully."
                            )
                        else:
                            st.success(
                                "All predictions have at least 70% confidence."
                            )

                        st.divider()
                        st.subheader("🔎 Filter Batch Results")

                        filter_col1, filter_col2 = st.columns(2)

                        with filter_col1:
                            prediction_filter = st.selectbox(
                                "Prediction",
                                ["All", "FAKE NEWS", "REAL NEWS"],
                                key="prediction_filter"
                            )

                        with filter_col2:
                            confidence_filter = st.selectbox(
                                "Confidence",
                                [
                                    "All",
                                    "High (90%+)",
                                    "Medium (70% - 89%)",
                                    "Low (<70%)"
                                ],
                                key="confidence_filter"
                            )

                        search_text = st.text_input(
                            "🔍 Search Articles",
                            placeholder="Enter a keyword...",
                            key="batch_search"
                        )

                        filtered_results = results.copy()

                        if prediction_filter != "All":
                            filtered_results = filtered_results[
                                filtered_results["Prediction"] == prediction_filter
                            ]

                        if confidence_filter == "High (90%+)":
                            filtered_results = filtered_results[
                                filtered_results["Confidence (%)"] >= 90
                            ]

                        elif confidence_filter == "Medium (70% - 89%)":
                            filtered_results = filtered_results[
                                (filtered_results["Confidence (%)"] >= 70)
                                &
                                (filtered_results["Confidence (%)"] < 90)
                            ]

                        elif confidence_filter == "Low (<70%)":
                            filtered_results = filtered_results[
                                filtered_results["Confidence (%)"] < 70
                            ]

                        if search_text.strip():
                            filtered_results = filtered_results[
                                filtered_results["Article"].str.contains(
                                    search_text,
                                    case=False,
                                    na=False
                                )
                            ]

                        st.write(
                            f"Showing **{len(filtered_results)}** "
                            f"of **{len(results)}** articles"
                        )

                        if filtered_results.empty:
                            st.warning(
                                "No articles match the selected filters."
                            )
                        else:
                            st.dataframe(
                                filtered_results,
                                use_container_width=True,
                                hide_index=True
                            )

                        st.download_button(
                            "📥 Download Filtered Batch Results",
                            filtered_results.to_csv(index=False).encode("utf-8"),
                            "batch_predictions.csv",
                            "text/csv",
                            key="download_batch_results"
                        )

        except Exception as e:

            st.error(
                "An error occurred while processing the CSV file."
            )

            st.exception(e)
