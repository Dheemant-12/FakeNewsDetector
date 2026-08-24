import os
import sys
import re
import json
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


# =================================================
# PROJECT ROOT
# =================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =================================================
# PROJECT IMPORTS
# =================================================

from model_metrics import load_metrics
from explainability.lime_explainer import explain_prediction
from utils.preprocessing import clean_text
from utils.tokenizer import tokenize_and_remove_stopwords
from batch_predict import predict_batch


# =================================================
# STREAMLIT CONFIGURATION
# =================================================

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)


# =================================================
# TEXT PREPROCESSING
# =================================================

def prepare_text(text):

    text = clean_text(text)

    text = tokenize_and_remove_stopwords(
        text
    )

    return text


# =================================================
# INPUT VALIDATION
# =================================================

def validate_input(text):

    text = text.strip()

    if not text:

        return (
            False,
            "Please enter a news article."
        )

    if not re.search(
        r"[A-Za-z]",
        text
    ):

        return (
            False,
            "Input should contain alphabetic text."
        )

    if len(text.split()) < 8:

        return (
            False,
            "Please enter a longer news article "
            "(at least 8 words)."
        )

    return True, ""


# =================================================
# LOAD MODEL
# =================================================

@st.cache_resource
def load_models():

    model = joblib.load(
        os.path.join(
            PROJECT_ROOT,
            "models",
            "logistic_regression.joblib"
        )
    )

    vectorizer = joblib.load(
        os.path.join(
            PROJECT_ROOT,
            "models",
            "tfidf_vectorizer.joblib"
        )
    )

    return model, vectorizer


model, vectorizer = load_models()


# =================================================
# LOAD MODEL METRICS
# =================================================

@st.cache_data
def get_model_metrics():

    return load_metrics()


try:

    metrics = get_model_metrics()

except Exception:

    metrics = None


# =================================================
# LOAD MODEL COMPARISON
# =================================================

@st.cache_data
def load_model_comparison():

    comparison_path = os.path.join(
        PROJECT_ROOT,
        "results",
        "model_comparison.json"
    )

    if not os.path.exists(
        comparison_path
    ):

        return None

    with open(
        comparison_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


try:

    model_comparison = load_model_comparison()

except Exception:

    model_comparison = None


# =================================================
# SESSION STATE
# =================================================

if "history" not in st.session_state:

    st.session_state.history = []


# =================================================
# SIDEBAR
# =================================================

st.sidebar.title(
    "📰 Fake News Detector"
)

st.sidebar.markdown("---")

st.sidebar.subheader(
    "👨‍💻 Developer"
)

st.sidebar.write(
    "Dheemant Reddy"
)

st.sidebar.subheader(
    "🤖 Model"
)

st.sidebar.write(
    "Logistic Regression"
)

st.sidebar.subheader(
    "📚 Vectorizer"
)

st.sidebar.write(
    "TF-IDF"
)

st.sidebar.subheader(
    "🛠 Tech Stack"
)

st.sidebar.markdown(
    """
- Python
- Streamlit
- Scikit-learn
- Pandas
- Joblib
- LIME
- Matplotlib
"""
)

st.sidebar.subheader(
    "📌 Version"
)

st.sidebar.write(
    "v1.0"
)

st.sidebar.markdown("---")

st.sidebar.info(
    "This project predicts whether a news "
    "article is Fake or Real using Machine Learning."
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Made with ❤️ using Streamlit"
)


# =================================================
# HEADER
# =================================================

st.title(
    "📰 Fake News Detector"
)

st.write(
    "Machine Learning powered Fake News "
    "Detection and Analysis System."
)


# =================================================
# NAVIGATION
# =================================================

tab_dashboard, tab_single, tab_batch = st.tabs(
    [
        "📊 Dashboard",
        "🔎 Single Prediction",
        "📂 Batch Prediction"
    ]
)
# =================================================
# DASHBOARD TAB
# =================================================

with tab_dashboard:

    st.header(
        "📊 Model Dashboard"
    )

    st.write(
        "Overview of the Fake News Detection "
        "model and its performance."
    )

    st.divider()


    # =================================================
    # MODEL INFORMATION
    # =================================================

    st.subheader(
        "🤖 Model Information"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Model",
            "Logistic Regression"
        )

    with col2:

        st.metric(
            "Vectorizer",
            "TF-IDF"
        )

    with col3:

        st.metric(
            "Features",
            "5000"
        )

    with col4:

        st.metric(
            "Classes",
            "Fake / Real"
        )


    st.info(
        "The application uses TF-IDF to convert "
        "news text into numerical features and "
        "Logistic Regression to classify the article."
    )


    # =================================================
    # HOW THE MODEL WORKS
    # =================================================

    with st.expander(
        "ℹ️ How does the model work?"
    ):

        st.markdown(
            """
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
"""
        )


    # =================================================
    # MODEL PERFORMANCE
    # =================================================

    st.divider()

    st.subheader(
        "📈 Model Performance"
    )

    if metrics is not None:

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

    else:

        st.warning(
            "Model performance metrics "
            "could not be loaded."
        )


    # =================================================
    # CLASSIFICATION SUMMARY
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

        summary_df["Score"] = (
            summary_df["Score"].round(4)
        )

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
    # PERFORMANCE INTERPRETATION
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


    # =================================================
    # CONFUSION MATRIX
    # =================================================

    if metrics is not None:

        st.divider()

        st.subheader(
            "🔲 Confusion Matrix"
        )

        matrix = metrics.get(
            "confusion_matrix"
        )

        if matrix:

            confusion_df = pd.DataFrame(
                matrix,
                index=[
                    "Actual Fake",
                    "Actual Real"
                ],
                columns=[
                    "Predicted Fake",
                    "Predicted Real"
                ]
            )

            st.dataframe(
                confusion_df,
                use_container_width=True
            )

        else:

            st.info(
                "Confusion matrix data "
                "is not available."
            )


    # =================================================
    # ROC CURVE
    # =================================================

    if metrics is not None:

        st.divider()

        st.subheader(
            "📈 ROC Curve"
        )

        roc_data = metrics.get(
            "roc_curve"
        )

        if roc_data:

            fpr = roc_data.get(
                "fpr",
                []
            )

            tpr = roc_data.get(
                "tpr",
                []
            )

            if fpr and tpr:

                fig, ax = plt.subplots(
                    figsize=(8, 5)
                )

                ax.plot(
                    fpr,
                    tpr,
                    label=(
                        f"AUC = "
                        f"{metrics['auc']:.4f}"
                    )
                )

                ax.plot(
                    [0, 1],
                    [0, 1],
                    linestyle="--",
                    label="Random Classifier"
                )

                ax.set_xlabel(
                    "False Positive Rate"
                )

                ax.set_ylabel(
                    "True Positive Rate"
                )

                ax.set_title(
                    "ROC Curve"
                )

                ax.legend()

                st.pyplot(fig)

                plt.close(fig)

            else:

                st.info(
                    "ROC curve data is empty."
                )

        else:

            st.info(
                "ROC curve data is not available."
            )


    # =================================================
    # MODEL COMPARISON
    # =================================================

    st.divider()

    st.subheader(
        "🤖 Model Comparison"
    )

    if model_comparison is not None:

        comparison_models = (
            model_comparison.get(
                "models",
                []
            )
        )

        best_model_name = (
            model_comparison.get(
                "best_model",
                "Unknown"
            )
        )

        if comparison_models:

            comparison_df = pd.DataFrame(
                comparison_models
            )


            # -----------------------------------------
            # BEST MODEL
            # -----------------------------------------

            st.success(
                f"🏆 Best Model: "
                f"{best_model_name}"
            )

            st.write(
                "The models below were trained using "
                "the same TF-IDF features and evaluated "
                "on the same test dataset."
            )


            # -----------------------------------------
            # COMPARISON TABLE
            # -----------------------------------------

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )


            # -----------------------------------------
            # ACCURACY COMPARISON
            # -----------------------------------------

            st.subheader(
                "📊 Accuracy Comparison"
            )

            accuracy_chart = comparison_df[
                [
                    "Model",
                    "Accuracy"
                ]
            ].copy()

            accuracy_chart = (
                accuracy_chart.set_index(
                    "Model"
                )
            )

            accuracy_chart["Accuracy"] = (
                accuracy_chart["Accuracy"] * 100
            )

            st.bar_chart(
                accuracy_chart
            )


            # -----------------------------------------
            # F1 SCORE COMPARISON
            # -----------------------------------------

            st.subheader(
                "🎯 F1 Score Comparison"
            )

            f1_chart = comparison_df[
                [
                    "Model",
                    "F1 Score"
                ]
            ].copy()

            f1_chart = (
                f1_chart.set_index(
                    "Model"
                )
            )

            f1_chart["F1 Score"] = (
                f1_chart["F1 Score"] * 100
            )

            st.bar_chart(
                f1_chart
            )


            # -----------------------------------------
            # OVERALL PERFORMANCE
            # -----------------------------------------

            st.subheader(
                "📈 Overall Model Performance"
            )

            performance_chart = comparison_df[
                [
                    "Model",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score"
                ]
            ].copy()

            performance_chart = (
                performance_chart.set_index(
                    "Model"
                )
            )

            performance_chart = (
                performance_chart * 100
            )

            st.bar_chart(
                performance_chart
            )


            # -----------------------------------------
            # MODEL SELECTION
            # -----------------------------------------

            st.subheader(
                "💡 Model Selection"
            )

            best_row = comparison_df[
                comparison_df["Model"]
                == best_model_name
            ]

            if not best_row.empty:

                best_f1 = best_row[
                    "F1 Score"
                ].iloc[0]

                best_accuracy = best_row[
                    "Accuracy"
                ].iloc[0]

                best_auc = best_row[
                    "AUC"
                ].iloc[0]

                st.info(
                    f"""
**{best_model_name}** achieved the highest
overall F1 Score in the model comparison.

- Accuracy: **{best_accuracy * 100:.2f}%**
- F1 Score: **{best_f1 * 100:.2f}%**
- AUC: **{best_auc:.4f}**

F1 Score balances precision and recall and
is useful when evaluating classification
performance.
"""
                )


            # -----------------------------------------
            # DOWNLOAD COMPARISON
            # -----------------------------------------

            comparison_csv = (
                comparison_df
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                label=(
                    "📥 Download Model Comparison"
                ),
                data=comparison_csv,
                file_name=(
                    "model_comparison.csv"
                ),
                mime="text/csv",
                key=(
                    "download_model_comparison"
                )
            )

        else:

            st.info(
                "No model comparison results found."
            )

    else:

        st.warning(
            "Model comparison results "
            "are not available."
        )

        st.info(
            "Run models/compare_models.py first."
        )
# =================================================
# SINGLE PREDICTION TAB
# =================================================

with tab_single:

    st.header(
        "🔎 Single Article Prediction"
    )

    st.write(
        "Enter one news article and let the "
        "model analyze it."
    )

    st.divider()


    # =================================================
    # ARTICLE INPUT
    # =================================================

    article = st.text_area(
        "Enter News Article",
        height=250,
        placeholder=(
            "Paste a news article here..."
        )
    )


    # =================================================
    # PREDICT BUTTON
    # =================================================

    if st.button(
        "🚀 Predict",
        key="single_predict"
    ):

        is_valid, message = validate_input(
            article
        )

        if not is_valid:

            st.warning(
                message
            )

        else:

            try:

                # =========================================
                # PREPROCESS TEXT
                # =========================================

                processed = prepare_text(
                    article
                )


                # =========================================
                # CREATE FEATURES
                # =========================================

                features = vectorizer.transform(
                    [processed]
                )


                # =========================================
                # MODEL PREDICTION
                # =========================================

                prediction = model.predict(
                    features
                )[0]


                probabilities = (
                    model.predict_proba(
                        features
                    )[0]
                )


                # =========================================
                # PROBABILITIES
                # =========================================

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


                # =========================================
                # CONFIDENCE LEVEL
                # =========================================

                if confidence >= 90:

                    confidence_level = (
                        "🟢 High"
                    )

                elif confidence >= 70:

                    confidence_level = (
                        "🟡 Medium"
                    )

                else:

                    confidence_level = (
                        "🔴 Low"
                    )


                # =================================================
                # PREDICTION RESULT
                # =================================================

                st.subheader(
                    "🎯 Prediction Result"
                )


                result_col1, result_col2 = (
                    st.columns([2, 1])
                )


                with result_col1:

                    if prediction == 0:

                        st.error(
                            "🚨 FAKE NEWS",
                            icon="🚨"
                        )

                        st.write(
                            "The model classified "
                            "this article as likely fake."
                        )

                    else:

                        st.success(
                            "✅ REAL NEWS",
                            icon="✅"
                        )

                        st.write(
                            "The model classified "
                            "this article as likely real."
                        )


                with result_col2:

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )


                # =================================================
                # PROBABILITY BREAKDOWN
                # =================================================

                st.subheader(
                    "📊 Probability Breakdown"
                )


                col1, col2 = st.columns(2)


                with col1:

                    st.metric(
                        "🚨 Fake Probability",
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
                        "✅ Real Probability",
                        f"{real_probability:.2f}%"
                    )

                    st.progress(
                        min(
                            int(real_probability),
                            100
                        )
                    )


                st.caption(
                    "These percentages represent the "
                    "model's estimated probability "
                    "for each class."
                )


                # =================================================
                # CONFIDENCE INTERPRETATION
                # =================================================

                st.subheader(
                    "🎯 Confidence Interpretation"
                )


                if confidence >= 90:

                    st.success(
                        f"🟢 High Confidence — "
                        f"{confidence:.2f}%"
                    )

                    confidence_message = (
                        "The model is highly confident "
                        "in this prediction."
                    )


                elif confidence >= 70:

                    st.warning(
                        f"🟡 Medium Confidence — "
                        f"{confidence:.2f}%"
                    )

                    confidence_message = (
                        "The model has reasonable confidence, "
                        "but the prediction should still "
                        "be reviewed."
                    )


                else:

                    st.error(
                        f"🔴 Low Confidence — "
                        f"{confidence:.2f}%"
                    )

                    confidence_message = (
                        "The model is uncertain about "
                        "this article. Do not rely on "
                        "this prediction alone."
                    )


                st.info(
                    confidence_message
                )


                # =================================================
                # MODEL DISCLAIMER
                # =================================================

                with st.expander(
                    "ℹ️ Important: How should this prediction be used?"
                ):

                    st.write(
                        "This system provides a machine-learning-based "
                        "prediction and should not be treated as a "
                        "definitive fact-checking system."
                    )

                    st.write(
                        "A high-confidence prediction does not "
                        "guarantee that an article is actually "
                        "fake or real."
                    )

                    st.write(
                        "For important claims, verify the information "
                        "using reliable sources and independent "
                        "fact-checking."
                    )


                # =================================================
                # DOWNLOAD PREDICTION REPORT
                # =================================================

                report = pd.DataFrame({

                    "Article": [
                        article
                    ],

                    "Prediction": [
                        (
                            "FAKE NEWS"
                            if prediction == 0
                            else "REAL NEWS"
                        )
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
                    ],

                    "Confidence (%)": [
                        round(
                            confidence,
                            2
                        )
                    ]
                })


                st.download_button(
                    label=(
                        "📥 Download Prediction Report"
                    ),

                    data=(
                        report
                        .to_csv(index=False)
                        .encode("utf-8")
                    ),

                    file_name=(
                        "prediction_report.csv"
                    ),

                    mime="text/csv",

                    key=(
                        "single_download"
                    )
                )


                # =================================================
                # LIME EXPLANATION
                # =================================================

                st.divider()

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


                # =================================================
                # LIME GRAPH
                # =================================================

                st.subheader(
                    "📊 Word Importance"
                )


                lime_df = lime_df.sort_values(
                    "Importance"
                )


                fig, ax = plt.subplots(
                    figsize=(8, 4)
                )


                ax.barh(
                    lime_df["Word"],
                    lime_df["Importance"]
                )


                ax.axvline(
                    0,
                    linewidth=1
                )


                ax.set_xlabel(
                    "Importance Score"
                )


                ax.set_title(
                    "Top Words Influencing Prediction"
                )


                st.pyplot(
                    fig
                )


                plt.close(
                    fig
                )


                # =================================================
                # IMPORTANT WORDS
                # =================================================

                st.subheader(
                    "🔍 Important Words"
                )


                important_words = (
                    processed.split()[:15]
                )


                if important_words:

                    st.write(
                        ", ".join(
                            important_words
                        )
                    )

                else:

                    st.write(
                        "No important words found."
                    )


                # =================================================
                # SAVE SESSION HISTORY
                # =================================================

                st.session_state.history.append({

                    "Time":
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),

                    "Prediction":
                        (
                            "FAKE NEWS"
                            if prediction == 0
                            else "REAL NEWS"
                        ),

                    "Confidence (%)":
                        round(
                            confidence,
                            2
                        ),

                    "Fake Probability (%)":
                        round(
                            fake_probability,
                            2
                        ),

                    "Real Probability (%)":
                        round(
                            real_probability,
                            2
                        ),

                    "Article":
                        (
                            article[:100] + "..."
                            if len(article) > 100
                            else article
                        )
                })


                # =================================================
                # DEBUG INFORMATION
                # =================================================

                with st.expander(
                    "🛠 Debug Information"
                ):

                    st.write(
                        "**Processed Text:**"
                    )

                    st.code(
                        processed
                    )

                    st.write(
                        f"**Feature Shape:** "
                        f"{features.shape}"
                    )

                    st.write(
                        f"**Non-zero Features:** "
                        f"{features.nnz}"
                    )


            except Exception as e:

                st.error(
                    "An unexpected error occurred "
                    "while making the prediction."
                )

                st.exception(
                    e
                )


    # =================================================
    # SESSION HISTORY & ANALYTICS
    # =================================================

    if st.session_state.history:

        st.divider()

        st.subheader(
            "📊 Session Analytics"
        )


        history_df = pd.DataFrame(
            st.session_state.history
        )


        # =================================================
        # BASIC STATISTICS
        # =================================================

        total_predictions = len(
            history_df
        )


        fake_predictions = (
            history_df["Prediction"]
            == "FAKE NEWS"
        ).sum()


        real_predictions = (
            history_df["Prediction"]
            == "REAL NEWS"
        ).sum()


        average_confidence = (
            history_df[
                "Confidence (%)"
            ].mean()
        )


        fake_percentage = (
            fake_predictions
            / total_predictions
            * 100
        )


        real_percentage = (
            real_predictions
            / total_predictions
            * 100
        )


        # =================================================
        # STATISTICS CARDS
        # =================================================

        col1, col2, col3, col4 = (
            st.columns(4)
        )


        with col1:

            st.metric(
                "Total Predictions",
                total_predictions
            )


        with col2:

            st.metric(
                "🚨 Fake",
                fake_predictions,
                f"{fake_percentage:.1f}%"
            )


        with col3:

            st.metric(
                "✅ Real",
                real_predictions,
                f"{real_percentage:.1f}%"
            )


        with col4:

            st.metric(
                "🎯 Avg Confidence",
                f"{average_confidence:.1f}%"
            )


        # =================================================
        # PREDICTION DISTRIBUTION
        # =================================================

        st.subheader(
            "📈 Prediction Distribution"
        )


        prediction_counts = pd.DataFrame(

            {
                "Predictions": [
                    fake_predictions,
                    real_predictions
                ]
            },

            index=[
                "FAKE NEWS",
                "REAL NEWS"
            ]
        )


        st.bar_chart(
            prediction_counts
        )


        # =================================================
        # CONFIDENCE OVERVIEW
        # =================================================

        st.subheader(
            "🎯 Confidence Overview"
        )


        confidence_data = history_df[
            [
                "Confidence (%)"
            ]
        ].copy()


        confidence_data.index = range(
            1,
            len(
                confidence_data
            ) + 1
        )


        confidence_data.index.name = (
            "Prediction"
        )


        st.line_chart(
            confidence_data
        )


        # =================================================
        # CONFIDENCE CATEGORIES
        # =================================================

        high_confidence = (
            history_df[
                "Confidence (%)"
            ] >= 90
        ).sum()


        medium_confidence = (
            (
                history_df[
                    "Confidence (%)"
                ] >= 70
            )
            &
            (
                history_df[
                    "Confidence (%)"
                ] < 90
            )
        ).sum()


        low_confidence = (
            history_df[
                "Confidence (%)"
            ] < 70
        ).sum()


        st.subheader(
            "🎯 Confidence Categories"
        )


        confidence_categories = pd.DataFrame(

            {
                "Predictions": [
                    high_confidence,
                    medium_confidence,
                    low_confidence
                ]
            },

            index=[
                "🟢 High (90%+)",
                "🟡 Medium (70–89%)",
                "🔴 Low (<70%)"
            ]
        )


        st.bar_chart(
            confidence_categories
        )


        # =================================================
        # LOW CONFIDENCE WARNING
        # =================================================

        if low_confidence > 0:

            st.warning(
                f"{low_confidence} prediction(s) "
                "have low confidence. These results "
                "should be reviewed carefully."
            )

        else:

            st.success(
                "All predictions have at least "
                "70% confidence."
            )


        # =================================================
        # DETAILED HISTORY
        # =================================================

        st.divider()

        st.subheader(
            "📜 Detailed Prediction History"
        )


        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # DOWNLOAD HISTORY
        # =================================================

        history_csv = (
            history_df
            .to_csv(index=False)
            .encode("utf-8")
        )


        st.download_button(
            label=(
                "📥 Download Prediction History"
            ),

            data=history_csv,

            file_name=(
                "prediction_history.csv"
            ),

            mime="text/csv",

            key=(
                "history_download"
            )
        )


        # =================================================
        # CLEAR HISTORY
        # =================================================

        if st.button(
            "🗑 Clear History",
            key="clear_history"
        ):

            st.session_state.history = []

            st.rerun()
        # =================================================
# BATCH PREDICTION TAB
# =================================================

with tab_batch:

    st.header(
        "📂 Batch News Prediction"
    )

    st.write(
        "Upload a CSV containing a column named "
        "'text' to predict multiple news articles."
    )

    st.divider()


    # =================================================
    # FILE UPLOAD
    # =================================================

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="batch_csv"
    )


    if uploaded_file is not None:

        try:

            # =============================================
            # READ CSV
            # =============================================

            batch_df = pd.read_csv(
                uploaded_file
            )


            # =============================================
            # VALIDATE COLUMN
            # =============================================

            if "text" not in batch_df.columns:

                st.error(
                    "❌ CSV must contain a "
                    "'text' column."
                )

            else:

                st.write(
                    f"📄 Articles found: "
                    f"{len(batch_df)}"
                )


                # =============================================
                # PREVIEW
                # =============================================

                with st.expander(
                    "👀 Preview Uploaded Data"
                ):

                    st.dataframe(
                        batch_df.head(10),
                        use_container_width=True,
                        hide_index=True
                    )


                # =============================================
                # BATCH PREDICTION BUTTON
                # =============================================

                if st.button(
                    "🚀 Predict Batch",
                    key="batch_predict"
                ):

                    results = predict_batch(
                        batch_df
                    )


                    # =========================================
                    # CHECK RESULTS
                    # =========================================

                    if results.empty:

                        st.warning(
                            "No valid articles were found "
                            "in the uploaded CSV."
                        )

                    else:

                        st.success(
                            f"Successfully predicted "
                            f"{len(results)} articles."
                        )


                        # =========================================
                        # BATCH ANALYTICS
                        # =========================================

                        st.subheader(
                            "📊 Batch Analytics Dashboard"
                        )


                        fake_count = (
                            results[
                                "Prediction"
                            ]
                            == "FAKE NEWS"
                        ).sum()


                        real_count = (
                            results[
                                "Prediction"
                            ]
                            == "REAL NEWS"
                        ).sum()


                        total_count = len(
                            results
                        )


                        average_confidence = (
                            results[
                                "Confidence (%)"
                            ].mean()
                        )


                        fake_percentage = (
                            fake_count
                            / total_count
                            * 100
                        )


                        real_percentage = (
                            real_count
                            / total_count
                            * 100
                        )


                        # =========================================
                        # STATISTICS
                        # =========================================

                        col1, col2, col3, col4 = (
                            st.columns(4)
                        )


                        with col1:

                            st.metric(
                                "Total Articles",
                                total_count
                            )


                        with col2:

                            st.metric(
                                "🚨 Fake News",
                                fake_count,
                                f"{fake_percentage:.1f}%"
                            )


                        with col3:

                            st.metric(
                                "✅ Real News",
                                real_count,
                                f"{real_percentage:.1f}%"
                            )


                        with col4:

                            st.metric(
                                "🎯 Avg Confidence",
                                f"{average_confidence:.1f}%"
                            )


                        # =========================================
                        # PREDICTION DISTRIBUTION
                        # =========================================

                        st.subheader(
                            "📈 Prediction Distribution"
                        )


                        prediction_chart = pd.DataFrame(

                            {
                                "Articles": [
                                    fake_count,
                                    real_count
                                ]
                            },

                            index=[
                                "FAKE NEWS",
                                "REAL NEWS"
                            ]
                        )


                        st.bar_chart(
                            prediction_chart
                        )


                        # =========================================
                        # CONFIDENCE DISTRIBUTION
                        # =========================================

                        st.subheader(
                            "🎯 Confidence Distribution"
                        )


                        confidence_chart = pd.DataFrame(

                            {
                                "Confidence (%)":
                                    results[
                                        "Confidence (%)"
                                    ].values
                            }
                        )


                        st.line_chart(
                            confidence_chart
                        )


                        # =========================================
                        # CONFIDENCE CATEGORIES
                        # =========================================

                        high_confidence = (
                            results[
                                "Confidence (%)"
                            ] >= 90
                        ).sum()


                        medium_confidence = (
                            (
                                results[
                                    "Confidence (%)"
                                ] >= 70
                            )
                            &
                            (
                                results[
                                    "Confidence (%)"
                                ] < 90
                            )
                        ).sum()


                        low_confidence = (
                            results[
                                "Confidence (%)"
                            ] < 70
                        ).sum()


                        st.subheader(
                            "🎯 Confidence Categories"
                        )


                        confidence_categories = pd.DataFrame(

                            {
                                "Articles": [
                                    high_confidence,
                                    medium_confidence,
                                    low_confidence
                                ]
                            },

                            index=[
                                "🟢 High (90%+)",
                                "🟡 Medium (70–89%)",
                                "🔴 Low (<70%)"
                            ]
                        )


                        st.bar_chart(
                            confidence_categories
                        )


                        # =========================================
                        # LOW CONFIDENCE WARNING
                        # =========================================

                        if low_confidence > 0:

                            st.warning(
                                f"{low_confidence} article(s) "
                                "have low prediction confidence. "
                                "Review them carefully."
                            )

                        else:

                            st.success(
                                "All predictions have at least "
                                "70% confidence."
                            )


                        # =========================================
                        # FILTERS
                        # =========================================

                        st.divider()

                        st.subheader(
                            "🔎 Filter Batch Results"
                        )


                        filter_col1, filter_col2 = (
                            st.columns(2)
                        )


                        # =========================================
                        # PREDICTION FILTER
                        # =========================================

                        with filter_col1:

                            prediction_filter = (
                                st.selectbox(
                                    "Prediction",

                                    [
                                        "All",
                                        "FAKE NEWS",
                                        "REAL NEWS"
                                    ],

                                    key=(
                                        "prediction_filter"
                                    )
                                )
                            )


                        # =========================================
                        # CONFIDENCE FILTER
                        # =========================================

                        with filter_col2:

                            confidence_filter = (
                                st.selectbox(
                                    "Confidence",

                                    [
                                        "All",
                                        "High (90%+)",
                                        "Medium (70% - 89%)",
                                        "Low (<70%)"
                                    ],

                                    key=(
                                        "confidence_filter"
                                    )
                                )
                            )


                        # =========================================
                        # SEARCH
                        # =========================================

                        search_text = st.text_input(
                            "🔍 Search Articles",
                            placeholder=(
                                "Enter a keyword..."
                            ),
                            key="batch_search"
                        )


                        # =========================================
                        # COPY RESULTS
                        # =========================================

                        filtered_results = (
                            results.copy()
                        )


                        # =========================================
                        # APPLY PREDICTION FILTER
                        # =========================================

                        if prediction_filter != "All":

                            filtered_results = (
                                filtered_results[
                                    filtered_results[
                                        "Prediction"
                                    ]
                                    == prediction_filter
                                ]
                            )


                        # =========================================
                        # APPLY CONFIDENCE FILTER
                        # =========================================

                        if (
                            confidence_filter
                            == "High (90%+)"
                        ):

                            filtered_results = (
                                filtered_results[
                                    filtered_results[
                                        "Confidence (%)"
                                    ] >= 90
                                ]
                            )


                        elif (
                            confidence_filter
                            == "Medium (70% - 89%)"
                        ):

                            filtered_results = (
                                filtered_results[
                                    (
                                        filtered_results[
                                            "Confidence (%)"
                                        ] >= 70
                                    )
                                    &
                                    (
                                        filtered_results[
                                            "Confidence (%)"
                                        ] < 90
                                    )
                                ]
                            )


                        elif (
                            confidence_filter
                            == "Low (<70%)"
                        ):

                            filtered_results = (
                                filtered_results[
                                    filtered_results[
                                        "Confidence (%)"
                                    ] < 70
                                ]
                            )


                        # =========================================
                        # APPLY SEARCH FILTER
                        # =========================================

                        if search_text.strip():

                            filtered_results = (
                                filtered_results[
                                    filtered_results[
                                        "Article"
                                    ]
                                    .astype(str)
                                    .str.contains(
                                        search_text,
                                        case=False,
                                        na=False
                                    )
                                ]
                            )


                        # =========================================
                        # FILTER SUMMARY
                        # =========================================

                        st.write(
                            f"Showing **"
                            f"{len(filtered_results)}"
                            f"** of **"
                            f"{len(results)}"
                            f"** articles"
                        )


                        # =========================================
                        # DISPLAY RESULTS
                        # =========================================

                        if filtered_results.empty:

                            st.warning(
                                "No articles match "
                                "the selected filters."
                            )

                        else:

                            st.dataframe(
                                filtered_results,
                                use_container_width=True,
                                hide_index=True
                            )


                        # =========================================
                        # DOWNLOAD RESULTS
                        # =========================================

                        filtered_csv = (
                            filtered_results
                            .to_csv(index=False)
                            .encode("utf-8")
                        )


                        st.download_button(
                            label=(
                                "📥 Download "
                                "Filtered Batch Results"
                            ),

                            data=filtered_csv,

                            file_name=(
                                "batch_predictions.csv"
                            ),

                            mime="text/csv",

                            key=(
                                "download_batch_results"
                            )
                        )


        # =========================================================
        # FILE PROCESSING ERROR
        # =========================================================

        except Exception as e:

            st.error(
                "An error occurred while "
                "processing the CSV file."
            )

            st.exception(
                e
            )