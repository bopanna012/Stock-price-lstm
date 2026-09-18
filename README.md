# Stock Price Prediction Using Multi-Layer LSTM

A time-series forecasting project that predicts next-day stock closing prices
from historical OHLCV (Open, High, Low, Close, Volume) data. Built as a
proper ML experiment rather than a single-model tutorial: a naive baseline
and a 1-layer LSTM are compared against a multi-layer (stacked) LSTM, using
chronological train/validation/test splitting and RMSE / MAE / MAPE
evaluation.

> **Status: work in progress.** This README reflects what has actually been
> built and verified so far, not the finished project.

## Why this project

A basic "download stock data → LSTM → plot" script is tutorial-level. This
project is designed to instead be a small forecasting *experiment*:
comparing models fairly, validating chronologically (no future-data leakage),
and being explicit about what the model can and can't actually tell you
about markets. See [Limitations](#limitations).

## Project structure

```
Stock_Price_Prediction/
├── config/
│   └── config.py           # all experiment settings in one place
├── data/
│   ├── raw/                 # downloaded OHLCV data (not committed, regenerated on demand)
│   └── processed/           # scaled sliding-window arrays (not committed, regenerated on demand)
├── models/                  # trained model weights + fitted scalers (not committed)
├── plots/                   # generated charts
├── results/
│   └── metrics.json         # RMSE/MAE/MAPE per model, for the Step 8 comparison
├── notebooks/                # exploratory notebooks
├── src/
│   ├── fetch_data.py        # Step 2: downloads OHLCV data via yfinance
│   ├── eda.py                # Step 3: exploratory data analysis
│   ├── preprocessing.py      # Step 4: scaling, sliding windows, chronological split
│   ├── metrics.py            # shared RMSE/MAE/MAPE functions
│   ├── results_store.py      # shared results/metrics.json read/write
│   └── baseline.py           # Step 5: naive baseline
├── requirements.txt          # top-level dependencies
├── requirements-lock.txt     # exact pinned versions (reproducibility)
└── .gitignore
```

## Setup

Requires Python 3.11.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

## How to run what exists so far

Run in order — each step depends on files produced by the previous one:

```bash
python src/fetch_data.py       # Step 2: downloads data/raw/AAPL.csv
python src/eda.py              # Step 3: prints data checks, saves plots/01-06*.png
python src/preprocessing.py    # Step 4: builds data/processed/{train,val,test}.npz + scalers
python src/baseline.py         # Step 5: naive baseline metrics + plots/07*.png
```

`fetch_data.py` pulls daily OHLCV history for the ticker/date range set in
`config/config.py` (currently AAPL, 2015-01-01 onward) using `yfinance`,
with prices adjusted for splits/dividends.

## Configuration

All experiment settings live in `config/config.py` instead of being
hardcoded in scripts:

| Setting | Current value | Meaning |
|---|---|---|
| `TICKER` | `AAPL` | Stock to model |
| `START_DATE` | `2015-01-01` | History start date |
| `FEATURE_COLUMNS` | `Open, High, Low, Close, Volume` | Model input features |
| `TARGET_COLUMN` | `Close` | What we're predicting |
| `LOOKBACK` | `60` | Trading days of history per input window |
| `HORIZON` | `1` | Predict this many days ahead (next close) |
| `TRAIN_FRACTION` / `VAL_FRACTION` / `TEST_FRACTION` | `0.70` / `0.15` / `0.15` | Chronological data split |
| `RANDOM_SEED` | `42` | Reproducibility |

## Exploratory data analysis (Step 3)

2,940 trading days of AAPL (2015-01-02 → present), zero missing values, zero
abnormal date gaps. Key findings (`src/eda.py`, plots in `plots/01-06*.png`):

- Price is up **1,274.6%** over the full period; average daily return
  **+0.106%**, daily volatility (std. dev.) **1.81%**.
- Largest single-day move: **+15.33%** (2025-04-09); largest single-day
  drop: **-12.86%** (2020-03-16, the COVID crash) — clearly visible as a
  spike in the rolling-volatility plot.
- **Close vs. Volume correlation ≈ -0.59** — counter-intuitive at first, but
  explained by AAPL's price growing ~14x while share count shrank from
  continuous buybacks, so fewer shares now move the same dollar volume.

## Preprocessing (Step 4)

`src/preprocessing.py` builds the model-ready sliding-window dataset:
chronological split (train = oldest 70%, val = next 15%, test = newest 15%,
no shuffling), `MinMaxScaler` fit **only** on training rows, then 60-day
input windows built across the full series and assigned to a split based on
where each window's *target* date falls (so val/test don't lose their first
60 rows to an artificial reset, while still never letting val/test data
influence training or the scaler).

Verified: 1,998 / 441 / 441 train/val/test windows, no NaNs, no chronological
overlap between splits.

**Notable finding:** because AAPL trended strongly upward, the train-fit
scaler puts **99.8% of test targets outside the [0,1] range** it was fit on
— the model has to extrapolate beyond any price it saw in training for
almost the entire test set. Documented here rather than hidden; see
[Limitations](#limitations).

## Results so far

| Model | Split | RMSE | MAE | MAPE |
|---|---|---|---|---|
| Naive baseline (tomorrow = today) | Test | $4.55 | $3.12 | 1.25% |

Naive baseline tracks actual price closely (`plots/07_baseline_test_predictions.png`)
— day-to-day closing-price changes are small relative to price level, so
"predict no change" already reaches ~1.25% test MAPE. **This is the bar the
LSTM models need to clear**; full comparison table lands in Step 8.

## Dataset

Source: [Yahoo Finance](https://finance.yahoo.com/) via the `yfinance`
Python package — free, no API key required. Currently AAPL, daily bars,
2015-01-02 through present (~2,940 trading days). Prices are adjusted for
stock splits and dividends (`auto_adjust=True`) so the series reflects real
value changes rather than accounting artifacts.

Raw data is **not committed to the repo** — it's regenerated by running
`src/fetch_data.py`, keeping the repo small and avoiding a stale, frozen
dataset.

## Planned next steps (not yet implemented)

- **1-layer LSTM** and a **stacked multi-layer LSTM** (with dropout between
  layers), trained with early stopping on validation loss.
- **Model comparison & error analysis:** RMSE/MAE/MAPE across naive baseline
  vs. 1-layer LSTM vs. multi-layer LSTM on the held-out test set, plus
  digging into where/when predictions go wrong.
- **Live data:** a Streamlit dashboard that pulls recent data via
  `yfinance` at inference time and shows the trained model's next-session
  prediction alongside current technical indicators.

## Limitations

This project forecasts a price series from its own historical
OHLCV patterns. It does not, and cannot, account for news, earnings,
macroeconomic conditions, interest rates, market sentiment, or unexpected
events — none of that is in the input data. Good error metrics on
historical test data are evidence of a working forecasting model; they are
**not** evidence of a profitable trading strategy. This project should be
read as a machine-learning time-series forecasting experiment, not a
trading system.
