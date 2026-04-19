# data_loader.py
# ETL: load, clean, and prepare price data

import pandas as pd
import numpy as np
from config import (
    DAILY_PRICES_PATH, MONTHLY_PRICES_PATH,
    DAILY_START_DATE, DOW_2003, DOW_FORWARD_STEPS,
    IQR_MULTIPLIER,
)


# ── Dow history ───────────────────────────────────────────────────────────────

def get_dow_forward_history() -> dict:
    """Return a dict {year: sorted list of tickers} from 2003 to 2026."""
    history = {}
    temp = list(DOW_2003)
    history[2003] = sorted(temp)

    for year in range(2004, 2027):
        if year in DOW_FORWARD_STEPS:
            for t in DOW_FORWARD_STEPS[year]["remove"]:
                if t in temp:
                    temp.remove(t)
            for t in DOW_FORWARD_STEPS[year]["add"]:
                temp.append(t)
        history[year] = sorted(list(temp))

    return history


# ── Load & clean ──────────────────────────────────────────────────────────────

def load_prices(path: str, start_date: str | None = None) -> pd.DataFrame:
    """
    Load a price CSV, drop columns with any NaN, and optionally
    filter to rows on or after *start_date*.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.dropna(axis=1, inplace=True)

    if start_date:
        df = df[df.index >= start_date].copy()

    return df


def load_daily_prices() -> pd.DataFrame:
    return load_prices(DAILY_PRICES_PATH, start_date=DAILY_START_DATE)


def load_monthly_prices() -> pd.DataFrame:
    return load_prices(MONTHLY_PRICES_PATH)


# ── Long-format helpers ───────────────────────────────────────────────────────

def prices_to_long_returns(prices: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Convert a wide price DataFrame to long-format monthly/daily returns
    suitable for StatsForecast.

    Returns columns: ds | unique_id | y
    """
    returns = prices.pct_change().dropna()
    df_long = (
        returns
        .reset_index()
        .melt(id_vars=date_col, var_name="unique_id", value_name="y")
        .rename(columns={date_col: "ds"})
        .reset_index(drop=True)
    )
    return df_long


# ── Outlier removal ───────────────────────────────────────────────────────────

def remove_outliers_iqr(
    df: pd.DataFrame,
    col: str = "y",
    multiplier: float = IQR_MULTIPLIER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove per-ticker outliers using the IQR method.

    Returns:
        df_clean    : DataFrame with outliers removed
        summary_df  : Summary of rows removed per ticker
    """
    results, summary = [], []

    for ticker in df["unique_id"].unique():
        sub = df[df["unique_id"] == ticker].copy()
        q1, q3 = sub[col].quantile(0.25), sub[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        mask = sub[col].between(lower, upper)

        summary.append({
            "Ticker":    ticker,
            "N_before":  len(sub),
            "N_removed": (~mask).sum(),
            "N_after":   mask.sum(),
            "Lower":     round(lower, 4),
            "Upper":     round(upper, 4),
        })
        results.append(sub[mask])

    df_clean   = pd.concat(results).reset_index(drop=True)
    summary_df = pd.DataFrame(summary).set_index("Ticker")

    print(f"Total rows removed: {summary_df['N_removed'].sum()} / {len(df)}")
    return df_clean, summary_df