# 📰 Fake News Detector

A machine-learning based web application for classifying news articles as **Fake News** or **Real News**.

The project combines text preprocessing, TF-IDF feature extraction, Logistic Regression, model comparison, LIME explainability, Streamlit visualization, session analytics, and batch prediction.

> **Final milestone status:** The current application is complete for this development cycle.
>
> **Important:** The trained model has known issues and limitations. It is not a production-grade or definitive fact-checking system. Anyone working with this repository is free to pull it, experiment with it, retrain the model, replace components, improve preprocessing, change the dataset, or redesign parts of the application.

---

## 🚀 Features

### 🔎 Single Article Prediction

- Article input and validation
- Text preprocessing
- TF-IDF feature extraction
- Fake/Real classification
- Fake and Real probabilities
- Confidence interpretation
- Downloadable prediction report

### 🧠 LIME Explainability

- Local explanation for individual predictions
- Important words
- Importance scores
- Word-importance visualization

### 📊 Model Evaluation

- Accuracy
- Precision
- Recall
- F1 Score
- AUC
- Confusion Matrix
- ROC Curve

### 🤖 Model Comparison

The project compares:

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM

Comparison metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- AUC

Results are stored in:

```text
results/model_comparison.csv
results/model_comparison.json
