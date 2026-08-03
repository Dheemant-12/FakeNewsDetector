import os
import sys

# -------------------------------------------------
# Add project root to Python path
# -------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from explainability.lime_explainer import explain_prediction

article = """
Scientists discovered that drinking ten cups of coffee
makes people immortal forever.
"""

exp = explain_prediction(article)

print("=" * 60)
print("LIME Explanation")
print("=" * 60)

for word, score in exp.as_list():
    print(f"{word:<20} {score:.4f}")