"""A tiny shared results registry (results/metrics.json).

Every model's evaluate step calls save_metrics() so that Step 8 (model
comparison) can just read one file instead of re-running every model.
"""

import json

from config.config import RESULTS_FILE


def save_metrics(model_name: str, metrics_by_split: dict) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results[model_name] = metrics_by_split

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)


def load_all_metrics() -> dict:
    if not RESULTS_FILE.exists():
        return {}
    with open(RESULTS_FILE) as f:
        return json.load(f)
