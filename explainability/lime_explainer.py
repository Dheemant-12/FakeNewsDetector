import os
import sys
import joblib
from lime.lime_text import LimeTextExplainer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

model = joblib.load("models/logistic_regression.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")


def predict_proba(texts):
    vectors = vectorizer.transform(texts)
    return model.predict_proba(vectors)


explainer = LimeTextExplainer(
    class_names=["Fake", "Real"]
)


def explain_prediction(text):

    return explainer.explain_instance(
        text,
        predict_proba,
        num_features=10
    )