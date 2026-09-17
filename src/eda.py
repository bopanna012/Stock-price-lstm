"""Step 3: Exploratory Data Analysis.

Loads the raw OHLCV CSV, checks basic data quality, and produces the
plots you'd want to look at before modeling: price trend, daily
returns, volume, rolling averages, and volatility. Nothing here
touches the LSTM yet -- this is purely "understand the data first".
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.config import TICKER, DATA_RAW_DIR, PLOTS_DIR


def load_data(ticker: str) -> pd.DataFrame:
    path = DATA_RAW_DIR / f"{ticker}.csv"
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df = df.sort_index()
    return df


def print_basic_checks(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("BASIC DATA CHECKS")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")

    print("\n--- First 5 rows ---")
    print(df.head())

    print("\n--- Data types ---")
    print(df.dtypes)

    print("\n--- Missing values per column ---")
    print(df.isnull().sum())

    print("\n--- Summary statistics ---")
    print(df.describe())

    # Trading calendars have gaps (weekends, holidays) -- that's normal.
    # A *large* gap could mean a data problem, so flag anything unusual.
    gaps = df.index.to_series().diff().dt.days
    large_gaps = gaps[gaps > 5]
    print(f"\n--- Gaps larger than 5 calendar days: {len(large_gaps)} ---")
    if len(large_gaps) > 0:
        print(large_gaps)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Daily return: how much the price moved vs. the previous day, in %.
    # This is what "volatility" and "risk" are usually measured from,
    # rather than raw price (a $1 move on a $10 stock is huge; on a
    # $1000 stock it's nothing).
    df["Daily_Return"] = df["Close"].pct_change() * 100

    # Rolling (moving) averages smooth out day-to-day noise so the
    # underlying trend is easier to see.
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["MA200"] = df["Close"].rolling(window=200).mean()

    # Rolling volatility: standard deviation of daily returns over a
    # trailing 20-day window. High = choppy/uncertain period.
    df["Volatility_20d"] = df["Daily_Return"].rolling(window=20).std()

    return df


def make_plots(df: pd.DataFrame, ticker: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Closing price with moving averages
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Close"], label="Close", linewidth=1)
    ax.plot(df.index, df["MA20"], label="20-day MA", linewidth=1)
    ax.plot(df.index, df["MA50"], label="50-day MA", linewidth=1)
    ax.plot(df.index, df["MA200"], label="200-day MA", linewidth=1)
    ax.set_title(f"{ticker} Closing Price with Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_closing_price_ma.png", dpi=150)
    plt.close(fig)

    # 2. Daily returns over time
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["Daily_Return"], linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{ticker} Daily Returns (%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return (%)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_daily_returns.png", dpi=150)
    plt.close(fig)

    # 3. Distribution of daily returns
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["Daily_Return"].dropna(), bins=100)
    ax.set_title(f"{ticker} Distribution of Daily Returns")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_returns_distribution.png", dpi=150)
    plt.close(fig)

    # 4. Trading volume over time
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["Volume"], linewidth=0.5, color="tab:orange")
    ax.set_title(f"{ticker} Trading Volume")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_volume.png", dpi=150)
    plt.close(fig)

    # 5. Rolling 20-day volatility
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["Volatility_20d"], linewidth=1, color="tab:red")
    ax.set_title(f"{ticker} 20-Day Rolling Volatility (Std. Dev. of Daily Returns)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (%)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "05_volatility.png", dpi=150)
    plt.close(fig)

    # 6. Correlation between OHLCV columns
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df[["Open", "High", "Low", "Close", "Volume"]].corr()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    ax.set_title(f"{ticker} OHLCV Correlation")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "06_correlation.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved 6 plots to {PLOTS_DIR}")


def print_insights(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)

    total_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    print(f"\nTotal price change over full period: {total_return:.1f}%")

    print(f"Average daily return: {df['Daily_Return'].mean():.3f}%")
    print(f"Daily return std dev (overall volatility): {df['Daily_Return'].std():.3f}%")

    biggest_gain = df["Daily_Return"].max()
    biggest_gain_date = df["Daily_Return"].idxmax().date()
    biggest_loss = df["Daily_Return"].min()
    biggest_loss_date = df["Daily_Return"].idxmin().date()
    print(f"Biggest single-day gain: {biggest_gain:.2f}% on {biggest_gain_date}")
    print(f"Biggest single-day loss: {biggest_loss:.2f}% on {biggest_loss_date}")

    corr_oc = df["Open"].corr(df["Close"])
    corr_cv = df["Close"].corr(df["Volume"])
    print(f"\nOpen vs Close correlation: {corr_oc:.3f} (expected: very high)")
    print(f"Close vs Volume correlation: {corr_cv:.3f} (usually weak/near zero)")


def main():
    df = load_data(TICKER)
    print_basic_checks(df)
    df = add_derived_columns(df)
    make_plots(df, TICKER)
    print_insights(df)


if __name__ == "__main__":
    main()
