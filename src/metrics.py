"""Shared evaluation metrics, used by every model (baseline + both LSTMs)
so comparisons in Step 8 are apples-to-apples.
"""

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error, in the target's original units ($).
    Penalizes large errors more than MAE (squares the error first).
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error, in $. Average size of the miss, no penalty
    weighting -- easiest metric to explain in plain English.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Expresses error relative to the
    price level, so it's comparable across very different price ranges
    (e.g. $20 AAPL in 2015 vs $330 AAPL in 2026).
    """
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }
