import os
import sys
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.train_logistic_regression import prepare_text
from utils.loader import load_dataset

from sklearn.model_selection import train_test_split

print("=" * 60)
print("Loading trained model...")
print("=" * 60)

model = joblib.load("models/logistic_regression.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

df = load_dataset()

df["processed_text"] = df["text"].apply(prepare_text)

X = df["processed_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

X_test = vectorizer.transform(X_test)

predictions = model.predict(X_test)

print("\nClassification Report\n")
print(classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions)

ConfusionMatrixDisplay(cm).plot()

plt.title("Confusion Matrix")

os.makedirs("results", exist_ok=True)

plt.savefig("results/confusion_matrix.png")

plt.close()

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X_test)[:, 1]
else:
    probabilities = model.decision_function(X_test)

fpr, tpr, _ = roc_curve(y_test, probabilities)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")

plt.plot([0,1],[0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

plt.savefig("results/roc_curve.png")

plt.close()

print(f"\nAUC Score : {roc_auc:.4f}")

print("\nEvaluation completed successfully.")