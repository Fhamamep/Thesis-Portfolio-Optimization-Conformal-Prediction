# config.py
# Central configuration for the pipeline

# ── Data ──────────────────────────────────────────────────────────────────────
DAILY_PRICES_PATH   = "data/dow_daily_prices.csv"
MONTHLY_PRICES_PATH = "data/dow_monthly_prices.csv"

DAILY_START_DATE    = "2013-06-01"   # Align with CP intervals

# ── Backtest ──────────────────────────────────────────────────────────────────
LOOKBACK_DAYS       = 756            # 3 years × 252 trading days
OPTIMIZE_EVERY      = 21             # Monthly rebalance (~21 trading days)
N_JOBS=-1

# ── Conformal Prediction ──────────────────────────────────────────────────────
CP_HORIZON          = 1              # 1-step-ahead forecast
CP_N_WINDOWS        = 36             # Calibration windows (3 years monthly)
CP_WINDOW_SIZE      = 36             # Rolling window for WindowAverage
CP_LEVELS           = [80, 85, 90, 95]
CP_ALPHAS           = [0.10, 0.15, 0.20]
CP_FREQ             = "ME"           # Month-end frequency

# ── Optimisation ──────────────────────────────────────────────────────────────
RISK_FREE_RATE      = 0.02
IQR_MULTIPLIER      = 1.5            # Outlier removal multiplier

# ── Dow 30 composition (2003 baseline) ───────────────────────────────────────
DOW_2003 = [
    "MMM", "AA", "MO", "AXP", "T",
    "BA",  "CAT", "C", "KO", "DD",
    "EK",  "XOM", "GE", "GM", "HPQ",
    "HD",  "HON", "INTC", "IBM", "IP",
    "JNJ", "JPM", "MCD", "MRK", "MSFT",
    "PG",  "SBC", "UTX", "WMT", "DIS",
]

DOW_FORWARD_STEPS = {
    2004: {"remove": ["T", "EK", "IP"],          "add": ["AIG", "PFE", "VZ"]},
    2005: {"remove": ["SBC"],                     "add": ["T"]},
    2008: {"remove": ["AIG", "MO", "HON"],        "add": ["KFT", "CVX", "BAC"]},
    2009: {"remove": ["GM", "C"],                 "add": ["TRV", "CSCO"]},
    2012: {"remove": ["KFT"],                     "add": ["UNH"]},
    2013: {"remove": ["AA", "BAC", "HPQ"],        "add": ["GS", "NKE", "V"]},
    2015: {"remove": ["T"],                       "add": ["AAPL"]},
    2017: {"remove": ["DD"],                      "add": ["DWDP"]},
    2018: {"remove": ["GE"],                      "add": ["WBA"]},
    2019: {"remove": ["DWDP"],                    "add": ["DOW"]},
    2020: {"remove": ["XOM", "PFE", "UTX"],       "add": ["AMGN", "HON", "CRM"]},
    2024: {"remove": ["INTC", "DOW", "WBA"],      "add": ["NVDA", "SHW", "AMZN"]},
}