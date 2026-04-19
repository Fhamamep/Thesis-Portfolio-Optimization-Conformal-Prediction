# metrics.py
# Portfolio performance and probabilistic evaluation metrics

from __future__ import annotations

from functools import partial
from typing import List

import numpy as np
import pandas as pd
from deel.puncc import metrics as puncc_metrics
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmse, mae, scaled_crps, mase, rmae, coverage


# ── Portfolio performance ─────────────────────────────────────────────────────

def portfolio_performance(returns: pd.DataFrame, rf_daily: float = 0.0) -> pd.DataFrame:
    """
    Compute annualised return, volatility, Sharpe ratio, and max drawdown.

    Args:
        returns   : DataFrame of daily portfolio returns
        rf_daily  : Daily risk-free rate

    Returns:
        DataFrame with metrics as rows, strategies as columns
    """
    trading_days = 252

    ann_ret = (1 + returns.mean()) ** trading_days - 1
    ann_vol = returns.std() * np.sqrt(trading_days)
    sharpe  = (returns.mean() - rf_daily) / returns.std() * np.sqrt(trading_days)

    def _max_drawdown(s: pd.Series) -> float:
        cum = (1 + s).cumprod()
        return ((cum - cum.cummax()) / cum.cummax()).min()

    mdd = returns.apply(_max_drawdown)

    return pd.DataFrame({
        "Annualised Return":    ann_ret,
        "Annualised Volatility": ann_vol,
        "Sharpe Ratio":         sharpe,
        "Maximum Drawdown":     mdd,
    })


# ── Forecast point metrics ────────────────────────────────────────────────────

def point_forecast_metrics(
    cv_df: pd.DataFrame,
    train_df: pd.DataFrame,
    models: List[str],
    levels: List[int],
) -> pd.DataFrame:
    """
    Compute RMSE, MAE, rMAE, and FVA for each model.

    Args:
        cv_df    : Cross-validation output (must have 'actual' column)
        train_df : Training data (must have 'actual' column)
        models   : List of model names
        levels   : Prediction interval levels

    Returns:
        Summary DataFrame indexed by model name
    """
    # Dummy cutoff so evaluate() treats all folds as one
    cv_copy = cv_df.copy()
    cv_copy["cutoff"] = pd.Timestamp("2099-01-01")

    mase_fn  = partial(mase, seasonality=1)
    rmae_fn  = partial(rmae, baseline="Naive")

    result = evaluate(
        cv_copy,
        train_df=train_df,
        metrics=[rmse, mae, scaled_crps, mase_fn, rmae_fn],
        level=levels,
        models=models,
        target_col="actual",
        agg_fn="mean",
    )

    rows = []
    for model in models:
        r = result[result["metric"].isin(["mae", "rmse", "rmae_Naive"])].set_index("metric")[model]
        rows.append({
            "Model":           model,
            "MAE":             r.get("mae",        np.nan),
            "RMSE":            r.get("rmse",       np.nan),
            "rMAE (vs Naive)": r.get("rmae_Naive", np.nan),
            "FVA":             1 - r.get("rmae_Naive", np.nan),
        })

    return pd.DataFrame(rows).set_index("Model").round(4)


# ── Probabilistic (interval) metrics ─────────────────────────────────────────

def interval_metrics(
    cv_df: pd.DataFrame,
    models: List[str],
    alphas: List[float],
) -> pd.DataFrame:
    """
    Compute marginal coverage, average width, and ACE for each
    (model, alpha) combination.

    Args:
        cv_df  : Cross-validation output with 'actual' column
        models : List of model names
        alphas : List of significance levels (e.g. [0.10, 0.15, 0.20])

    Returns:
        DataFrame indexed by (Model, Alpha)
    """
    rows = []

    for alpha in alphas:
        level = int((1 - alpha) * 100)

        for model in models:
            lo_col = f"{model}-lo-{level}"
            hi_col = f"{model}-hi-{level}"
            df = cv_df[["actual", lo_col, hi_col]].dropna()

            if df.empty:
                continue

            cov_val = puncc_metrics.regression_mean_coverage(
                df["actual"], df[lo_col], df[hi_col]
            )
            width_val = puncc_metrics.regression_sharpness(
                y_pred_lower=df[lo_col], y_pred_upper=df[hi_col]
            )
            ace_val = puncc_metrics.regression_ace(
                df["actual"], df[lo_col], df[hi_col], alpha
            )

            rows.append({
                "Model":     model,
                "Alpha":     alpha,
                "Level":     f"{level}%",
                "Coverage":  round(cov_val,       4),
                "Width %":   round(width_val * 100, 2),
                "ACE %":     round(ace_val   * 100, 2),
                "N":         len(df),
            })

    return pd.DataFrame(rows).set_index(["Model", "Alpha"])