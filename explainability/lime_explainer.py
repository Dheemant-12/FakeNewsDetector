import joblib
from lime.lime_text import LimeTextExplainer

# Load model
model = joblib.load("models/logistic_regression.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")


def predict_proba(texts):
    """
    LIME expects a prediction probability function.
    """

    vectors = vectorizer.transform(texts)

    return model.predict_proba(vectors)


explainer = LimeTextExplainer(
    class_names=["Fake", "Real"]
)


def explain_prediction(text):

    explanation = explainer.explain_instance(
        text,
        predict_proba,
        num_features=10
    )

    return explanation