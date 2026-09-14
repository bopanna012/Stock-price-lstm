"""Step 2: obtain a real OHLCV dataset via yfinance.

Downloads daily Open/High/Low/Close/Volume history for TICKER and
saves it as a plain CSV under data/raw/. Re-running this script always
re-downloads the latest history, so the raw data is treated as
regenerable and is not committed to git (see .gitignore).
"""

import sys
from pathlib import Path

import yfinance as yf

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import TICKER, START_DATE, END_DATE, INTERVAL, DATA_RAW_DIR


def fetch_ohlcv(ticker: str, start: str, end: str | None, interval: str):
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,   # adjusts Close for splits/dividends
        progress=False,
    )

    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check the ticker/date range.")

    # yfinance can return MultiIndex columns (e.g. ('Close', 'AAPL'));
    # flatten to plain OHLCV column names for a single ticker.
    if isinstance(df.columns, __import__("pandas").MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    return df


def main():
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_ohlcv(TICKER, START_DATE, END_DATE, INTERVAL)

    out_path = DATA_RAW_DIR / f"{TICKER}.csv"
    df.to_csv(out_path)

    print(f"Saved {len(df)} rows for {TICKER} -> {out_path}")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(df.head())


if __name__ == "__main__":
    main()
