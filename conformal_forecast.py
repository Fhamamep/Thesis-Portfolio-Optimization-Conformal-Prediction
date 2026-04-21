# conformal_forecast.py  
# Conformal prediction of monthly returns — run independently from main pipeline.  
#  
# USAGE:  
#   python conformal_forecast.py  
#  
# Outputs saved to:  
#   outputs/cp_lower_bounds.csv  
#   outputs/point_forecast_metrics.csv  
#   outputs/interval_forecast_metrics.csv  
  
from __future__ import annotations  
  
import os  
import warnings  
import matplotlib.pyplot as plt  
import numpy as np  
from functools import partial  
  
import pandas as pd  
  
warnings.filterwarnings("ignore")  
  
from statsforecast import StatsForecast  
from statsforecast.models import (  
    WindowAverage,  
    Naive,  
    RandomWalkWithDrift,  
    HistoricAverage,  
    AutoETS,
    AutoARIMA
)  
from statsforecast.utils import ConformalIntervals  
  
from config import (  
    CP_HORIZON, CP_N_WINDOWS, CP_WINDOW_SIZE,  
    CP_LEVELS, CP_ALPHAS, CP_FREQ, N_JOBS,  
)  
from data_loader import (  
    load_monthly_prices, prices_to_long_returns, remove_outliers_iqr,  
)  
from metrics import point_forecast_metrics, interval_metrics  
  
  
# ─────────────────────────────────────────────────────────────────────────────  
# MODEL BUILDER  
# ─────────────────────────────────────────────────────────────────────────────  
  
def build_models(  
    horizon: int     = CP_HORIZON,  
    n_windows: int   = CP_N_WINDOWS,  
    window_size: int = CP_WINDOW_SIZE,  
) -> list:  
    """Instantiate StatsForecast models with conformal intervals."""  
    intervals = ConformalIntervals(h=horizon, n_windows=n_windows)  
    return [  
        WindowAverage(window_size=window_size, prediction_intervals=intervals),  
        Naive(prediction_intervals=intervals),  
        RandomWalkWithDrift(prediction_intervals=intervals),  
        HistoricAverage(prediction_intervals=intervals),  
        AutoETS(prediction_intervals=intervals),  
        AutoARIMA(prediction_intervals=intervals),  
    ]  
  
  
# ─────────────────────────────────────────────────────────────────────────────  
# CROSS VALIDATION  
# NOTE: Not included in main.py pipeline — run this file independently.  
# Running CV inside the backtest loop would repeat the full CV at every  
# rebalance step, which is computationally prohibitive.  
# ─────────────────────────────────────────────────────────────────────────────  
  
def run_cross_validation(  
    train_long: pd.DataFrame,  
    horizon: int          = CP_HORIZON,  
    n_windows: int | None = None,  
    levels: list          = CP_LEVELS,  
    freq: str             = CP_FREQ,  
    n_jobs: int           = N_JOBS,  
) -> pd.DataFrame:  
    """  
    Run StatsForecast cross-validation on long-format return data.  
  
    Args:  
        train_long : DataFrame with columns [ds, unique_id, y]  
        horizon    : Forecast horizon (periods)  
        n_windows  : Number of CV windows; auto-computed if None  
        levels     : Prediction interval levels  
        freq       : Pandas frequency string  
        n_jobs     : Parallel jobs (1 on Windows, -1 on Linux/macOS)  
  
    Returns:  
        cv_df : Cross-validation results DataFrame  
    """  
    n_obs = train_long.groupby("unique_id")["ds"].count().min()  
  
    if n_windows is None:  
        n_windows = (n_obs - CP_WINDOW_SIZE - horizon) // horizon  
        print(f"  Auto n_windows = {n_windows}")  
  
    models = build_models(horizon=horizon, n_windows=CP_N_WINDOWS)  
  
    sf = StatsForecast(  
        models=models,  
        freq=freq,  
        n_jobs=n_jobs,  
        verbose=True,  
    )  
  
    cv_df = sf.cross_validation(  
        df=train_long,  
        h=horizon,  
        step_size=horizon,  
        n_windows=n_windows,  
        level=levels,  
    )  
    return cv_df  # cross validation df per ticker


  
  
# ─────────────────────────────────────────────────────────────────────────────  
# CP BOUND EXTRACTION  
# ─────────────────────────────────────────────────────────────────────────────  
  
def extract_cp_bounds(  
    cv_df: pd.DataFrame,  
    model: str = "WindowAverage",  
    level: int = 80,  
) -> pd.DataFrame:  
    """  
    Pivot CV results to a wide DataFrame of lower-bound forecasts.  
  
    Args:  
        cv_df  : Output of run_cross_validation  
        model  : Model name (e.g. "WindowAverage")  
        level  : Confidence level (e.g. 80 for the 80% lower bound)  
  
    Returns:  
        Wide DataFrame indexed by ds, columns = tickers  
    """  
    col = f"{model}-lo-{level}"  
    wide = cv_df.pivot(index="ds", columns="unique_id", values=col)  
    wide.dropna(inplace=True)  
    return wide  
  
  
# ─────────────────────────────────────────────────────────────────────────────  
# STANDALONE PIPELINE  
# ─────────────────────────────────────────────────────────────────────────────  
  
def run_conformal_pipeline() -> None:  
    """  
    Standalone conformal prediction pipeline.  
    Runs CV, evaluates forecast metrics, and saves all outputs.  
    """  
    os.makedirs("outputs", exist_ok=True)  
  
    # ── Load and clean data ───────────────────────────────────────────────────  
    print("=" * 70)  
    print("Loading monthly prices")  
    print("=" * 70)  
  
    monthly_prices     = load_monthly_prices()  
    monthly_long_raw   = prices_to_long_returns(monthly_prices, date_col="Date")  
    monthly_long_clean, _ = remove_outliers_iqr(monthly_long_raw, col="y")  
  
    # ── Cross-validation ──────────────────────────────────────────────────────  
    print("\n" + "=" * 70)  
    print("Running cross-validation")  
    print("=" * 70)  
  
    cv_df = run_cross_validation(  
        train_long=monthly_long_clean,  
        levels=CP_LEVELS,
        n_windows=CP_N_WINDOWS, 
    )  
  
    cv_df              = cv_df.rename(columns={"y": "actual"})  
    monthly_long_clean = monthly_long_clean.rename(columns={"y": "actual"})  
  
    # ── Extract CP lower bounds ───────────────────────────────────────────────  
    print("\n" + "=" * 70)  
    print("Extracting CP lower bounds")  
    print("=" * 70)  
  
    cp_returns = extract_cp_bounds(cv_df, model="WindowAverage", level=80)  
    print(f"CP bounds shape: {cp_returns.shape}")  
  
    # ── Point forecast metrics ────────────────────────────────────────────────  
    print("\n" + "=" * 70)  
    print("Point forecast metrics (MAE, RMSE, rMAE, FVA)")  
    print("=" * 70)  
  
    models = ["WindowAverage", "Naive", "RWD", "HistoricAverage"]  
  
    point_metrics_df = point_forecast_metrics(  
        cv_df=cv_df,  
        train_df=monthly_long_clean,  
        models=models,  
        levels=CP_LEVELS,  
    )  
    print(point_metrics_df.to_string())  
  
    # ── Interval forecast metrics ─────────────────────────────────────────────  
    print("\n" + "=" * 70)  
    print("Probabilistic interval metrics (Coverage, Width, ACE)")  
    print("=" * 70)  
  
    interval_metrics_df = interval_metrics(  
        cv_df=cv_df,  
        models=models,  
        alphas=CP_ALPHAS,  
    )  
    print(interval_metrics_df.to_string())  
  
    # ── Save outputs ──────────────────────────────────────────────────────────  
    print("\n" + "=" * 70)  
    print("Saving outputs")  
    print("=" * 70)  
  
    cp_returns.to_csv("outputs/cp_lower_bounds.csv")  
    point_metrics_df.to_csv("outputs/point_forecast_metrics.csv")  
    interval_metrics_df.to_csv("outputs/interval_forecast_metrics.csv")  
  
    print("Saved:")  
    print("  outputs/cp_lower_bounds.csv")  
    print("  outputs/point_forecast_metrics.csv")  
    print("  outputs/interval_forecast_metrics.csv")  
    analyse_lower_bounds(cv_df, levels=CP_LEVELS)  
    print("\nConformal pipeline complete.")  
  
  
# ─────────────────────────────────────────────────────────────────────────────  
# ENTRY POINT  
# ─────────────────────────────────────────────────────────────────────────────  
# ─────────────────────────────────────────────────────────────────────────────  
# LOWER BOUND ANALYSIS  
# ─────────────────────────────────────────────────────────────────────────────  
  # conformal_forecast.py    

def analyse_lower_bounds(cv_df: pd.DataFrame, levels: list = CP_LEVELS) -> None:  
    """  
    Descriptive analysis of CP lower bound values across all tickers,  
    models and confidence levels.  
  
    Answers the key question:  
        Are the lower bounds positive enough to be used as MVO expected  
        return inputs, or are they predominantly negative?  
  
    Args:  
        cv_df  : Cross-validation output with 'actual' column  
        levels : Confidence levels to analyse (e.g. [80, 85, 90, 95])  
    """  
  
    models = ["WindowAverage", "Naive", "RWD", "HistoricAverage"]  
  
    # ── 1. Global descriptive stats per (model, level) ────────────────────────  
    print("\n" + "=" * 70)  
    print("LOWER BOUND ANALYSIS")  
    print("=" * 70)  
  
    print("\n[1] Global descriptive statistics of lower bounds")  
    print("-" * 70)  
  
    summary_rows = []  
  
    for model in models:  
        for level in levels:  
            col = f"{model}-lo-{level}"  
            if col not in cv_df.columns:  
                continue  
  
            series = cv_df[col].dropna()  
            n_total    = len(series)  
            n_negative = (series < 0).sum()  
            n_positive = (series >= 0).sum()  
            pct_neg    = n_negative / n_total * 100  
  
            summary_rows.append({  
                "Model":        model,  
                "Level":        f"{level}%",  
                "Mean":         round(series.mean(), 4),  
                "Std":          round(series.std(), 4),  
                "Min":          round(series.min(), 4),  
                "25%":          round(series.quantile(0.25), 4),  
                "Median":       round(series.median(), 4),  
                "75%":          round(series.quantile(0.75), 4),  
                "Max":          round(series.max(), 4),  
                "N Negative":   n_negative,  
                "N Positive":   n_positive,  
                "% Negative":   round(pct_neg, 1),  
            })  
  
    summary_df = pd.DataFrame(summary_rows).set_index(["Model", "Level"])  
    print(summary_df.to_string())  
  
    # ── 2. Per-ticker sign analysis (WindowAverage, level 80) ─────────────────  
    print("\n[2] Per-ticker sign analysis — WindowAverage lo-80")  
    print("-" * 70)  
    print("(Most relevant for MVO since lo-80 is the least conservative bound)\n")  
  
    col = "WindowAverage-lo-80"  
    if col in cv_df.columns:  
        ticker_sign = (  
            cv_df.groupby("unique_id")[col]  
            .apply(lambda s: pd.Series({  
                "Mean":       round(s.mean(), 4),  
                "Min":        round(s.min(), 4),  
                "Max":        round(s.max(), 4),  
                "% Negative": round((s < 0).mean() * 100, 1),  
                "% Positive": round((s >= 0).mean() * 100, 1),  
                "N":          s.dropna().__len__(),  
            }))  
            .unstack()  
        )  
        print(ticker_sign.to_string())  
  
        # Summary sentence  
        all_neg_tickers = ticker_sign[ticker_sign["% Negative"] == 100].index.tolist()  
        mixed_tickers   = ticker_sign[  
            (ticker_sign["% Negative"] > 0) & (ticker_sign["% Negative"] < 100)  
        ].index.tolist()  
  
        print(f"\n  Tickers with 100% negative lower bounds : {len(all_neg_tickers)}")  
        print(f"  Tickers with mixed sign lower bounds    : {len(mixed_tickers)}")  
        print(f"  List of all-negative tickers: {all_neg_tickers}")  
  
    # ── 3. Comparison: lower bound vs actual mean return ──────────────────────  
    print("\n[3] Lower bound vs actual mean return — WindowAverage lo-80")  
    print("-" * 70)  
  
    col = "WindowAverage-lo-80"  
    if col in cv_df.columns and "actual" in cv_df.columns:  
        comparison = (  
            cv_df.groupby("unique_id")  
            .apply(lambda g: pd.Series({  
                "Actual Mean Return": round(g["actual"].mean(), 4),  
                "CP Lower Bound Mean": round(g[col].mean(), 4),  
                "Difference":         round(g["actual"].mean() - g[col].mean(), 4),  
                "LB Usable for MVO":  "Yes" if g[col].mean() > 0 else "No",  
            }))  
        )  
        print(comparison.to_string())  
  
        n_usable = (comparison["LB Usable for MVO"] == "Yes").sum()  
        print(f"\n  Tickers where mean lower bound > 0 (usable for MVO): "  
              f"{n_usable} / {len(comparison)}")  
  
    # ── 4. Distribution plot — lower bounds across all levels ─────────────────  
    print("\n[4] Generating distribution plots...")  
  
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  
    axes = axes.flatten()  
  
    for i, model in enumerate(models):  
        ax = axes[i]  
        for level in levels:  
            col = f"{model}-lo-{level}"  
            if col in cv_df.columns:  
                cv_df[col].dropna().plot.kde(ax=ax, label=f"lo-{level}")  
  
        ax.axvline(0, color="red", linewidth=1.2, linestyle="--", label="Zero")  
        ax.set_title(f"{model} — Lower Bound Distribution", fontweight="bold")  
        ax.set_xlabel("Lower Bound Value")  
        ax.set_ylabel("Density")  
        ax.legend(fontsize=8)  
        ax.grid(True, alpha=0.3)  
  
    plt.suptitle(  
        "CP Lower Bound Distributions by Model and Confidence Level\n"  
        "(red dashed line = zero; values left of zero not usable as MVO inputs)",  
        fontsize=11, fontweight="bold", y=1.01,  
    )  
    plt.tight_layout()  
    plt.savefig("outputs/plots/cp_lower_bound_distributions.png", dpi=150, bbox_inches="tight")  
    print("  Saved: outputs/plots/cp_lower_bound_distributions.png")  
    plt.show()  
  
    # ── 5. Heatmap — % negative lower bounds per ticker (WindowAverage) ───────  
    print("\n[5] Generating heatmap of % negative lower bounds...")  
  
    heatmap_data = pd.DataFrame()  
    for level in levels:  
        col = f"WindowAverage-lo-{level}"  
        if col in cv_df.columns:  
            pct_neg = cv_df.groupby("unique_id")[col].apply(  
                lambda s: (s < 0).mean() * 100  
            )  
            heatmap_data[f"lo-{level}"] = pct_neg  
  
    if not heatmap_data.empty:  
        fig, ax = plt.subplots(figsize=(8, 12))  
        im = ax.imshow(heatmap_data.values, aspect="auto", cmap="RdYlGn_r",  
                       vmin=0, vmax=100)  
  
        ax.set_xticks(range(len(heatmap_data.columns)))  
        ax.set_xticklabels(heatmap_data.columns)  
        ax.set_yticks(range(len(heatmap_data.index)))  
        ax.set_yticklabels(heatmap_data.index)  
  
        # Annotate cells  
        for row in range(len(heatmap_data.index)):  
            for col_idx in range(len(heatmap_data.columns)):  
                val = heatmap_data.values[row, col_idx]  
                ax.text(col_idx, row, f"{val:.0f}%",  
                        ha="center", va="center", fontsize=8,  
                        color="black")  
  
        plt.colorbar(im, ax=ax, label="% Negative Lower Bounds")  
        ax.set_title(  
            "WindowAverage: % of Lower Bound Values < 0\nper Ticker and Confidence Level",  
            fontweight="bold",  
        )  
        plt.tight_layout()  
        plt.savefig("outputs/plots/cp_lower_bound_heatmap.png", dpi=150, bbox_inches="tight")  
        print("  Saved: outputs/plots/cp_lower_bound_heatmap.png")  
        plt.show()  
  
    # ── 6. Conclusion ─────────────────────────────────────────────────────────  
    print("\n" + "=" * 70)  
    print("CONCLUSION")  
    print("=" * 70)  
    print("""  
  CP lower bounds represent the worst-case expected return at a given  
  confidence level. By construction they are conservative and will be  
  negative for most tickers in most periods.  
  
  Implications for portfolio optimisation:  
  ─────────────────────────────────────────  
  1. DIRECT MVO INPUT: Not suitable. MVO max-Sharpe requires at least  
     one asset with expected return > risk-free rate. Negative bounds  
     will cause optimisation failures (as seen in the backtest output).  
  
  2. RISK FILTER: Suitable. Use the lower bound to screen OUT assets  
     whose worst-case return falls below a threshold (e.g. < -5%),  
     then run standard MVO on the remaining universe.  
  
  3. CONSERVATIVE ALLOCATION: Suitable. Use the lower bound as the  
     expected return input to a min-variance or risk-parity optimiser  
     that does not require positive expected returns.  
  
  4. SCENARIO ANALYSIS: Suitable. Compare MVO results using point  
     forecasts vs lower bounds to stress-test portfolio robustness.  
    """)  

if __name__ == "__main__":  
    try:  
        run_conformal_pipeline()  
    except KeyboardInterrupt:  
        print("\nStopped by user.")  