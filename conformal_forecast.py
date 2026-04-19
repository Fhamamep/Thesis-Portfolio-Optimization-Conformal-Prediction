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
from functools import partial  
  
import pandas as pd  
  
warnings.filterwarnings("ignore")  
  
from statsforecast import StatsForecast  
from statsforecast.models import (  
    WindowAverage,  
    Naive,  
    RandomWalkWithDrift,  
    HistoricAverage,  
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
    return cv_df  
  
  
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
    print("\nConformal pipeline complete.")  
  
  
# ─────────────────────────────────────────────────────────────────────────────  
# ENTRY POINT  
# ─────────────────────────────────────────────────────────────────────────────  
  
if __name__ == "__main__":  
    try:  
        run_conformal_pipeline()  
    except KeyboardInterrupt:  
        print("\nStopped by user.")  