"""Step 5: Naive baseline.

The naive forecast: tomorrow's closing price = today's closing price.
No learning involved at all. This exists to answer one question that
every later model has to beat: "is the LSTM actually adding value, or
could a straight line do just as well?"

Convenient side effect of how we built the sliding windows in Step 4:
the last time step of each input window IS "today" (the day right
before the target), so the naive prediction is simply the Close value
at position [-1] of each window -- no extra data loading needed.
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import (
    TICKER,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    FEATURE_COLUMNS,
)
from metrics import evaluate
from results_store import save_metrics

CLOSE_IDX = FEATURE_COLUMNS.index("Close")


def load_split(name: str):
    data = np.load(DATA_PROCESSED_DIR / f"{name}.npz")
    return data["X"], data["y"], data["target_row_idx"]


def naive_predict(X: np.ndarray) -> np.ndarray:
    """Scaled prediction for each window = the Close value on the last
    day of the window (i.e. "today"), since HORIZON=1.
    """
    return X[:, -1, CLOSE_IDX]


def main():
    target_scaler = joblib.load(MODELS_DIR / "target_scaler.joblib")
    df = pd.read_csv(DATA_RAW_DIR / f"{TICKER}.csv", index_col="Date", parse_dates=True).sort_index()

    print("=" * 60)
    print("NAIVE BASELINE: tomorrow's close = today's close")
    print("=" * 60)

    all_metrics = {}
    test_dates = test_actual = test_pred = None

    for split in ["train", "val", "test"]:
        X, y_scaled, target_row_idx = load_split(split)

        pred_scaled = naive_predict(X)

        # Inverse-transform back to real dollars for interpretable metrics.
        y_true = target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
        y_pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

        m = evaluate(y_true, y_pred)
        all_metrics[split] = m
        print(f"\n{split.upper()}  (n={len(y_true)})")
        print(f"  RMSE: ${m['rmse']:.2f}")
        print(f"  MAE:  ${m['mae']:.2f}")
        print(f"  MAPE: {m['mape']:.2f}%")

        if split == "test":
            test_dates = df.index[target_row_idx]
            test_actual, test_pred = y_true, y_pred

    save_metrics("naive_baseline", all_metrics)
    print(f"\nSaved metrics to results/metrics.json under 'naive_baseline'")

    # Plot: actual vs. naive-predicted close on the test set.
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_dates, test_actual, label="Actual Close", linewidth=1)
    ax.plot(test_dates, test_pred, label="Naive Prediction (= previous day)", linewidth=1, linestyle="--")
    ax.set_title(f"{TICKER} Naive Baseline -- Test Set")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "07_baseline_test_predictions.png", dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {PLOTS_DIR / '07_baseline_test_predictions.png'}")


if __name__ == "__main__":
    main()
