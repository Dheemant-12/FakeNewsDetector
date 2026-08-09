import json
import os


# -------------------------------------------------
# Metrics File
# -------------------------------------------------

METRICS_PATH = os.path.join(
    "models",
    "metrics.json"
)


# -------------------------------------------------
# Load Saved Metrics
# -------------------------------------------------

def load_metrics():

    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(
            "models/metrics.json was not found."
        )

    with open(
        METRICS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        metrics = json.load(file)

    return metrics