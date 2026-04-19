# main.py  
# Orchestrates the full pipeline end-to-end  
  
import warnings  
import os  
  
import pandas as pd  
import matplotlib.pyplot as plt  
  
warnings.filterwarnings("ignore")  
  
from config import (  
    LOOKBACK_DAYS, OPTIMIZE_EVERY,  
    CP_LEVELS,  
    RISK_FREE_RATE
)  
from data_loader import (  
    load_daily_prices, load_monthly_prices,  
    prices_to_long_returns, remove_outliers_iqr,  
)  
from optimizers import equal_weight, max_sharpe, min_volatility, hrp  
from backtest import backtest  
from metrics import portfolio_performance  
  
  
# ─────────────────────────────────────────────────────────────────────────────  
# PLOT HELPERS  
# ─────────────────────────────────────────────────────────────────────────────  
  
def plot_cumulative_returns(  
    returns: pd.DataFrame,  
    title: str = "Cumulative Returns",  
    save_path: str | None = None,  
) -> None:  
    """Plot cumulative returns for all strategies."""  
    fig, ax = plt.subplots(figsize=(12, 6))  
    for col in returns.columns:  
        (1 + returns[col]).cumprod().plot(ax=ax, label=col)  
    ax.axhline(1, color="black", linewidth=0.8, linestyle="--", alpha=0.4)  
    ax.set_title(title, fontsize=13, fontweight="bold")  
    ax.set_ylabel("Cumulative Return (rebased to 1)")  
    ax.set_xlabel("Date")  
    ax.legend(framealpha=0.9)  
    ax.grid(True, alpha=0.3)  
    plt.tight_layout()  
    if save_path:  
        fig.savefig(save_path, dpi=150)  
        print(f"  Saved: {save_path}")  
    plt.show()  
  
  
def plot_rolling_sharpe(  
    returns: pd.DataFrame,  
    window: int = 252,  
    rf_daily: float = 0.0,  
    title: str = "Rolling 1-Year Sharpe Ratio",  
    save_path: str | None = None,  
) -> None:  
    """Plot rolling Sharpe ratio for all strategies."""  
    fig, ax = plt.subplots(figsize=(12, 5))  
    for col in returns.columns:  
        excess = returns[col] - rf_daily  
        roll_sharpe = (  
            excess.rolling(window).mean()  
            / excess.rolling(window).std()  
            * (252 ** 0.5)  
        )  
        roll_sharpe.plot(ax=ax, label=col)  
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)  
    ax.set_title(title, fontsize=13, fontweight="bold")  
    ax.set_ylabel("Sharpe Ratio")  
    ax.set_xlabel("Date")  
    ax.legend(framealpha=0.9)  
    ax.grid(True, alpha=0.3)  
    plt.tight_layout()  
    if save_path:  
        fig.savefig(save_path, dpi=150)  
        print(f"  Saved: {save_path}")  
    plt.show()  
  
  
def plot_drawdowns(  
    returns: pd.DataFrame,  
    title: str = "Drawdowns",  
    save_path: str | None = None,  
) -> None:  
    """Plot drawdown series for all strategies."""  
    fig, ax = plt.subplots(figsize=(12, 5))  
    for col in returns.columns:  
        cum = (1 + returns[col]).cumprod()  
        drawdown = (cum - cum.cummax()) / cum.cummax()  
        drawdown.plot(ax=ax, label=col)  
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)  
    ax.set_title(title, fontsize=13, fontweight="bold")  
    ax.set_ylabel("Drawdown")  
    ax.set_xlabel("Date")  
    ax.legend(framealpha=0.9)  
    ax.grid(True, alpha=0.3)  
    plt.tight_layout()  
    if save_path:  
        fig.savefig(save_path, dpi=150)  
        print(f"  Saved: {save_path}")  
    plt.show()  
  
  
def plot_weight_evolution(  
    weights_dict: dict,  
    strategy: str,  
    top_n: int = 5,  
    save_path: str | None = None,  
) -> None:  
    """Plot weight evolution for the top-N most-held assets of a strategy."""  
    if strategy not in weights_dict:  
        print(f"  Strategy '{strategy}' not found in weights_dict.")  
        return  
    w = weights_dict[strategy].astype(float)  
    top_assets = w.abs().mean().nlargest(top_n).index  
    fig, ax = plt.subplots(figsize=(12, 5))  
    w[top_assets].plot(ax=ax)  
    ax.set_title(  
        f"Weight Evolution – {strategy} (top {top_n} assets)",  
        fontsize=13, fontweight="bold",  
    )  
    ax.set_ylabel("Portfolio Weight")  
    ax.set_xlabel("Date")  
    ax.legend(framealpha=0.9)  
    ax.grid(True, alpha=0.3)  
    plt.tight_layout()  
    if save_path:  
        fig.savefig(save_path, dpi=150)  
        print(f"  Saved: {save_path}")  
    plt.show()  
  
  
# ─────────────────────────────────────────────────────────────────────────────  
# MAIN PIPELINE  
# ─────────────────────────────────────────────────────────────────────────────  
  
def main() -> None:  
  
    os.makedirs("outputs/plots", exist_ok=True)  
  
    # ── Step 1: Load data ─────────────────────────────────────────────────────  
    print("=" * 70)  
    print("STEP 1: Loading data")  
    print("=" * 70)  
  
    daily_prices   = load_daily_prices()  
    #monthly_prices = load_monthly_prices()  
  
    print(f"Daily prices  : {daily_prices.shape}  "  
          f"({daily_prices.index[0].date()} → {daily_prices.index[-1].date()})")  
   # print(f"Monthly prices: {monthly_prices.shape}  "  
    #      f"({monthly_prices.index[0].date()} → {monthly_prices.index[-1].date()})")  
  
    # ── Step 2: Conformal prediction (disabled — run separately) ──────────────  
    # NOTE: CP is computationally expensive and is run independently via  
    # conformal_forecast.py. The resulting cp_lower_bounds.csv is loaded  
    # in the CP-enabled backtest variant when ready.  
    #  
    # monthly_long_raw   = prices_to_long_returns(monthly_prices, date_col="Date")  
    # monthly_long_clean, _ = remove_outliers_iqr(monthly_long_raw, col="y")  
    # cv_df              = run_cross_validation(train_long=monthly_long_clean, levels=CP_LEVELS)  
    # cv_df              = cv_df.rename(columns={"y": "actual"})  
    # monthly_long_clean = monthly_long_clean.rename(columns={"y": "actual"})  
    # cp_returns         = extract_cp_bounds(cv_df, model="WindowAverage", level=80)  
    # print(f"CP bounds shape: {cp_returns.shape}")  
  
    # ── Step 3: Rolling backtest ──────────────────────────────────────────────  
    print("\n" + "=" * 70)  
    print("STEP 3: Walk-forward backtest (rolling window)")  
    print("=" * 70)  
  
    portfolio_functions = {  
        "MaxSharpe":   max_sharpe,  
        "MinVol":      min_volatility,  
        "HRP":         hrp,  
        "EqualWeight": equal_weight,  
        # "CP_MaxSharpe": lambda X: cp_max_sharpe(X, cp_returns),  # re-enable when CP is ready  
    }  
  
    returns_df, weights_dict = backtest(  
        portfolio_funcs=portfolio_functions,  
        prices=daily_prices,  
        lookback=LOOKBACK_DAYS,  
        optimize_every=OPTIMIZE_EVERY,  
        window_type="rolling",  
        cost_bps=10,  
        verbose=True,  
    )  
  
    # ── Expanding window backtest (disabled) ───────────────────────────────────  
    # returns_df_exp, _ = backtest(  
    #     portfolio_funcs=portfolio_functions,  
    #     prices=daily_prices,  
    #     lookback=LOOKBACK_DAYS,  
    #     optimize_every=OPTIMIZE_EVERY,  
    #     window_type="expanding",  
    #     cost_bps=10,  
    #     verbose=True,  
    # )  
  
    # ── Step 4: Portfolio performance metrics ─────────────────────────────────  
    print("\n" + "=" * 70)  
    print("STEP 4: Portfolio performance metrics")  
    print("=" * 70)  
  
    rf_daily     = (1 + RISK_FREE_RATE) ** (1 / 252) - 1  #compounding annual risk-free rate to daily
    perf_rolling = portfolio_performance(returns_df, rf_daily=rf_daily)  
  
    print("\nRolling window:")  
    print(perf_rolling.round(4).to_string())  
   
  
    # ── Step 5: Save outputs ──────────────────────────────────────────────────  
    print("\n" + "=" * 70)  
    print("STEP 5: Saving outputs")  
    print("=" * 70)  
  
    returns_df.to_csv("outputs/portfolio_returns_rolling.csv")  
    perf_rolling.to_csv("outputs/performance_rolling.csv")  
  
    # Disabled until CP pipeline is re-enabled:  
    # perf_expanding.to_csv("outputs/performance_expanding.csv")  
    # point_metrics_df.to_csv("outputs/point_forecast_metrics.csv")  
    # interval_metrics_df.to_csv("outputs/interval_forecast_metrics.csv")  
    # cp_returns.to_csv("outputs/cp_lower_bounds.csv")  
  
    print("All outputs saved to outputs/")  
  
    # ── Step 7: Plots ─────────────────────────────────────────────────────────  
    print("\n" + "=" * 70)  
    print("STEP 7: Plots")  
    print("=" * 70)  
  
    plot_cumulative_returns(  
        returns_df,  
        title="Cumulative Returns – Rolling Window",  
        save_path="outputs/plots/cumulative_rolling.png",
    )  
    plot_rolling_sharpe(  
        returns_df,  
        window=252,  
        rf_daily=rf_daily,  
        title="Rolling 1-Year Sharpe – Rolling Window",  
        save_path="outputs/plots/sharpe_rolling.png",  
    )  
    plot_drawdowns(  
        returns_df,  
        title="Drawdowns – Rolling Window",  
        save_path="outputs/plots/drawdowns_rolling.png",  
    )  
    plot_weight_evolution(  
        weights_dict,  
        strategy="MaxSharpe",  
        top_n=5,  
        save_path="outputs/plots/weights_maxsharpe.png",  
    )  
  
    # Disabled until CP and expanding window are re-enabled:  
    # plot_cumulative_returns(returns_df_exp, title="Cumulative Returns – Expanding Window", ...)  
    # plot_weight_evolution(weights_dict, strategy="CP_MaxSharpe", ...)  
  
    print("\nPipeline complete.")  
  
  
# ─────────────────────────────────────────────────────────────────────────────  
# ENTRY POINT  
# ─────────────────────────────────────────────────────────────────────────────  
  
if __name__ == "__main__":  
    try:  
        main()  
    except KeyboardInterrupt:  
        print("\nStopped by user.")  