# backtest.py
# Walk-forward backtester supporting rolling and expanding windows

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd


def _run_strategy(
    portfolio_name: str,
    portfolio_func: Callable,
    X: pd.DataFrame,
    lookback: int,
    optimize_every: int,
    window_type: str,          # "rolling" | "expanding"
    cost_bps: float,
    verbose: bool,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Core loop for a single strategy."""
    N = X.shape[1]
    current_w = np.zeros(N)    # weights initialised to zero

    w_store   = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    ret_store = pd.Series(index=X.index, dtype=float, name=portfolio_name)

    T = len(X)

    if verbose:
        print(f"\\nBacktesting: {portfolio_name}  [{window_type} window]")
        print(f"  Periods: {T}  |  Lookback: {lookback}  |  OOS: {T - lookback}")

    for t in range(lookback, T):
        reoptimise = (t - lookback) % optimize_every == 0

        if reoptimise:
            if window_type == "rolling":
                train = X.iloc[t - lookback : t]
            else:                                   # expanding
                train = X.iloc[0 : t]

            new_w = portfolio_func(train)

            if new_w is not None:
                turnover = float(np.abs(new_w - current_w).sum())
                tc = turnover * cost_bps / 1e4
                current_w = new_w
            else:
                tc = 0.0

            if verbose:
                date_str = X.index[t].date()
                print(f"  t={t} ({date_str})  train={train.index[0].date()}→{train.index[-1].date()}")
        else:
            tc = 0.0

        # Record weights and period return
        w_store.iloc[t]   = current_w
        period_ret        = float((X.iloc[t] * current_w).sum()) - tc
        ret_store.iloc[t] = period_ret

        # Weight drift
        current_w = current_w * (1.0 + X.iloc[t].values)
        total = current_w.sum()
        if total > 0:
            current_w /= total

    return ret_store[lookback:], w_store[lookback:]


def backtest(
    portfolio_funcs: Dict[str, Callable],
    prices: pd.DataFrame,
    lookback: int,
    optimize_every: int = 1,
    window_type: str = "rolling",
    cost_bps: float = 0.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Walk-forward backtester for multiple portfolio strategies.

    Args:
        portfolio_funcs : {name: callable(train_returns) -> np.ndarray}
        prices          : Wide price DataFrame (index=dates, columns=tickers)
        lookback        : Training window length in periods
        optimize_every  : Re-optimisation frequency in periods
        window_type     : "rolling" (fixed size) or "expanding" (growing)
        cost_bps        : Transaction cost in basis points per unit of turnover
        verbose         : Print progress

    Returns:
        (returns_df, weights_dict)
    """
    X = prices.pct_change().dropna()

    all_rets, all_ws = {}, {}

    for name, func in portfolio_funcs.items():
        rets, ws = _run_strategy(
            portfolio_name=name,
            portfolio_func=func,
            X=X,
            lookback=lookback,
            optimize_every=optimize_every,
            window_type=window_type,
            cost_bps=cost_bps,
            verbose=verbose,
        )
        all_rets[name] = rets
        all_ws[name]   = ws

        if verbose:
            ann_ret = (1 + rets).prod() ** (252 / len(rets)) - 1
            ann_vol = rets.std() * np.sqrt(252)
            sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0
            print(f"  ✓ Ann.Return={ann_ret:.2%}  Ann.Vol={ann_vol:.2%}  Sharpe={sharpe:.3f}")

    return pd.DataFrame(all_rets), all_ws