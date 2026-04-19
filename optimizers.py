# optimizers.py
# Portfolio optimisation functions used inside the backtester

import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import HRPOpt
from config import RISK_FREE_RATE


def equal_weight(train_returns: pd.DataFrame) -> np.ndarray:
    """Equal-weight (1/N) portfolio."""
    n = train_returns.shape[1]
    return np.ones(n) / n


def max_sharpe(
    train_returns: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> np.ndarray:
    """
    Max-Sharpe MVO using annualised returns and daily covariance.
    Falls back to equal weight if optimisation fails.
    """
    cov = train_returns.cov()
    ann_ret = (1 + train_returns.mean()) ** 252 - 1

    try:
        ef = EfficientFrontier(ann_ret, cov)
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        return np.array(list(ef.clean_weights().values()))
    except ValueError as exc:
        print(f"  [max_sharpe] Optimisation failed: {exc}. Falling back to equal weight.")
        return equal_weight(train_returns)


def min_volatility(train_returns: pd.DataFrame) -> np.ndarray:
    """
    Minimum-variance MVO.
    Falls back to equal weight if optimisation fails.
    """
    cov = train_returns.cov()
    ann_ret = (1 + train_returns.mean()) ** 252 - 1

    try:
        ef = EfficientFrontier(ann_ret, cov)
        ef.min_volatility()
        return np.array(list(ef.clean_weights().values()))
    except ValueError as exc:
        print(f"  [min_volatility] Optimisation failed: {exc}. Falling back to equal weight.")
        return equal_weight(train_returns)


def hrp(train_returns: pd.DataFrame) -> np.ndarray:
    """Hierarchical Risk Parity (HRP)."""
    opt = HRPOpt(train_returns)
    opt.optimize()
    return np.array(list(opt.clean_weights().values()))


def cp_max_sharpe(
    train_returns: pd.DataFrame,
    cp_returns: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> np.ndarray:
    """
    Max-Sharpe MVO where expected returns come from the conformal
    prediction lower bound for the current month.

    Args:
        train_returns : Daily returns for the training window (for covariance).
        cp_returns    : Wide-format DataFrame of CP lower bounds indexed by month.
    """
    last_period = train_returns.index.max().to_period("M")
    cp_month = cp_returns[cp_returns.index.to_period("M") == last_period]

    # Drop any tickers not present in train_returns
    common = [c for c in cp_month.columns if c in train_returns.columns]
    cp_month = cp_month[common]
    train_sub = train_returns[common]

    cov = train_sub.cov()
    ann_ret = (1 + cp_month.mean()) ** 12 - 1   # monthly → annual

    try:
        ef = EfficientFrontier(ann_ret, cov)
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        weights_dict = ef.clean_weights()
        # Re-align to full ticker list
        w = np.array([weights_dict.get(t, 0.0) for t in train_returns.columns])
        return w
    except ValueError as exc:
        print(f"  [cp_max_sharpe] Optimisation failed: {exc}. Falling back to equal weight.")
        return equal_weight(train_returns)