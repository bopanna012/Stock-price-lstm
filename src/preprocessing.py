"""Step 4: Time-series preprocessing.

Turns the raw OHLCV CSV into model-ready arrays:
    1. Chronological train / validation / test split (no shuffling).
    2. Scaling (MinMaxScaler) fit ONLY on the training portion.
    3. Sliding-window sequences: LOOKBACK past days -> next day's Close.

The output of this script is what every model (naive baseline, 1-layer
LSTM, multi-layer LSTM) will train and be evaluated on, so getting it
right here matters more than any individual model.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import (
    TICKER,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    LOOKBACK,
    HORIZON,
    TRAIN_FRACTION,
    VAL_FRACTION,
)


def load_data(ticker: str) -> pd.DataFrame:
    path = DATA_RAW_DIR / f"{ticker}.csv"
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df.sort_index()


def chronological_split_bounds(n_rows: int) -> tuple[int, int]:
    """Return (train_end, val_end) row indices for a 70/15/15-style split.

    Rows [0, train_end) are train, [train_end, val_end) are val,
    [val_end, n_rows) are test. Purely index-based (no shuffling) so
    order in time is preserved.
    """
    train_end = int(n_rows * TRAIN_FRACTION)
    val_end = int(n_rows * (TRAIN_FRACTION + VAL_FRACTION))
    return train_end, val_end


def fit_scalers(df: pd.DataFrame, train_end: int):
    """Fit scalers using ONLY the training rows.

    Two scalers, both fit on the same training rows:
      - feature_scaler: all OHLCV columns -> used as model input.
      - target_scaler: Close only -> lets us cheaply inverse-transform
        a predicted Close back to dollars later, without needing to
        reconstruct a full fake OHLCV row just to invert one column.

    This is one of the most important correctness rules in time-series
    ML: if the scaler saw validation/test values while fitting, it has
    effectively "seen the future" (e.g. it knows the true min/max price
    range that only occurs later), which leaks information the model
    wouldn't really have at prediction time.
    """
    train_df = df.iloc[:train_end]

    feature_scaler = MinMaxScaler()
    feature_scaler.fit(train_df[FEATURE_COLUMNS])

    target_scaler = MinMaxScaler()
    target_scaler.fit(train_df[[TARGET_COLUMN]])

    return feature_scaler, target_scaler


def build_sequences(scaled_features: np.ndarray, scaled_target: np.ndarray):
    """Slide a LOOKBACK-day window across the *entire* scaled series.

    For each position i, X is scaled_features[i : i+LOOKBACK] and y is
    scaled_target[i+LOOKBACK+HORIZON-1] (the close HORIZON days after
    the window ends). Returns X, y, and target_row_idx -- the original
    row index each y value came from, so we can later decide which
    split (train/val/test) each window belongs to based on where its
    *target* falls, and also recover real dates for plotting.
    """
    X, y, target_row_idx = [], [], []
    n = len(scaled_features)

    last_start = n - LOOKBACK - HORIZON  # last valid window start
    for i in range(last_start + 1):
        window_end = i + LOOKBACK
        target_i = window_end + HORIZON - 1
        X.append(scaled_features[i:window_end])
        y.append(scaled_target[target_i])
        target_row_idx.append(target_i)

    return np.array(X), np.array(y), np.array(target_row_idx)


def split_sequences(X, y, target_row_idx, train_end, val_end):
    """Assign each window to train/val/test based on its TARGET row.

    A window can legitimately reach back across a split boundary (e.g.
    a validation window's first few days are technically from the
    training period) -- that's fine and standard: it's still real past
    data, and no val/test label ever influenced training or the
    scaler. What matters is that each window's *target* (what we're
    scoring the model on) only appears in one split.
    """
    train_mask = target_row_idx < train_end
    val_mask = (target_row_idx >= train_end) & (target_row_idx < val_end)
    test_mask = target_row_idx >= val_end

    splits = {}
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        splits[name] = {
            "X": X[mask],
            "y": y[mask],
            "target_row_idx": target_row_idx[mask],
        }
    return splits


def sanity_checks(df, splits, train_end, val_end):
    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    for name in ["train", "val", "test"]:
        X, y = splits[name]["X"], splits[name]["y"]
        print(f"\n{name}: X shape = {X.shape}, y shape = {y.shape}")
        assert not np.isnan(X).any(), f"{name} X contains NaNs"
        assert not np.isnan(y).any(), f"{name} y contains NaNs"

    # Confirm chronological ordering: every train target date precedes
    # every val target date, which precedes every test target date.
    train_dates = df.index[splits["train"]["target_row_idx"]]
    val_dates = df.index[splits["val"]["target_row_idx"]]
    test_dates = df.index[splits["test"]["target_row_idx"]]

    print(f"\nTrain target dates: {train_dates.min().date()} to {train_dates.max().date()}")
    print(f"Val   target dates: {val_dates.min().date()} to {val_dates.max().date()}")
    print(f"Test  target dates: {test_dates.min().date()} to {test_dates.max().date()}")

    assert train_dates.max() < val_dates.min(), "Train/val overlap in time!"
    assert val_dates.max() < test_dates.min(), "Val/test overlap in time!"
    print("\nNo chronological overlap between splits. Good.")

    # A real, important caveat for stock data: since AAPL's price has
    # trended strongly upward, the training period's min/max Close is
    # very likely lower than what appears later. MinMaxScaler fit only
    # on train will then produce val/test values outside [0, 1]. That's
    # not a bug -- it's an honest signal that the model is being asked
    # to extrapolate beyond the price range it trained on.
    val_y = splits["val"]["y"]
    test_y = splits["test"]["y"]
    val_out_of_range = ((val_y < 0) | (val_y > 1)).mean() * 100
    test_out_of_range = ((test_y < 0) | (test_y > 1)).mean() * 100
    print(f"\nVal targets outside [0,1] scaled range:  {val_out_of_range:.1f}%")
    print(f"Test targets outside [0,1] scaled range: {test_out_of_range:.1f}%")
    if test_out_of_range > 0:
        print(
            "-> Expected for a stock in a long-term uptrend: the model must "
            "extrapolate beyond the price range it was trained on. Worth "
            "calling out explicitly in the final error analysis."
        )


def main():
    df = load_data(TICKER)
    n = len(df)
    train_end, val_end = chronological_split_bounds(n)

    feature_scaler, target_scaler = fit_scalers(df, train_end)

    scaled_features = feature_scaler.transform(df[FEATURE_COLUMNS])
    scaled_target = target_scaler.transform(df[[TARGET_COLUMN]]).ravel()

    X, y, target_row_idx = build_sequences(scaled_features, scaled_target)
    splits = split_sequences(X, y, target_row_idx, train_end, val_end)

    sanity_checks(df, splits, train_end, val_end)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name in ["train", "val", "test"]:
        np.savez(
            DATA_PROCESSED_DIR / f"{name}.npz",
            X=splits[name]["X"],
            y=splits[name]["y"],
            target_row_idx=splits[name]["target_row_idx"],
        )

    joblib.dump(feature_scaler, MODELS_DIR / "feature_scaler.joblib")
    joblib.dump(target_scaler, MODELS_DIR / "target_scaler.joblib")

    print(f"\nSaved train/val/test .npz files -> {DATA_PROCESSED_DIR}")
    print(f"Saved feature_scaler.joblib and target_scaler.joblib -> {MODELS_DIR}")


if __name__ == "__main__":
    main()
