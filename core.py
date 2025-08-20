# core.py

"""
Pure, testable helpers for the Stock Price Dashboard.
No Streamlit, no caching — safe to unit test.

Provided utilities
    Data:
    - load_prices(tickers_list, start_date, end_date, download_fn)

    Returns:
    - log_returns(price_df)

    Rebasing / comparison:
    - normalize_for_comparison(prices_df)      # common-date rebase to 100
    - rebase_to_index(series_or_df, base=1.0)  # quick rebase to 1.0 or 100

    Indicators / risk:
    - moving_averages(price_series, windows)
    - rolling_volatility(returns, window, annualize=True, periods_per_year=252)
    - compute_drawdown(price_series) -> (drawdown_series, max_drawdown)
"""

from __future__ import annotations
from typing import Callable, Sequence, Any, Iterable

import numpy as np
import pandas as pd
import yfinance as yf

__all__ = [
    "load_prices",
    "log_returns",
    "normalize_for_comparison",
    "rebase_to_index",
    "moving_averages",
    "rolling_volatility",
    "compute_drawdown",
]

# --------------------------- Data --------------------------------------------

def load_prices(
    tickers_list: Sequence[str],
    start_date: Any,
    end_date: Any,
    download_fn: Callable[..., pd.DataFrame] = yf.download,
) -> pd.DataFrame:
    """
    Download **adjusted Close** prices for one or more tickers over a date range.

    Returns a DataFrame: index=dates, columns=tickers (float).
    Columns entirely missing are dropped. Empty if no "Close" found or no tickers.
    """
    if not tickers_list:
        return pd.DataFrame()

    df = download_fn(
        tickers=tickers_list,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )
    if "Close" not in df:
        return pd.DataFrame()

    close = df["Close"].copy()
    if isinstance(close, pd.Series):  # single ticker normalization
        close = close.to_frame(tickers_list[0])
    return close.dropna(how="all")


# --------------------------- Returns / Rebasing -------------------------------

def log_returns(price_df: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Log (continuous) returns: r_t = ln(P_t / P_{t-1}).
    Same shape as input; first row NaN.
    """
    return np.log(price_df / price_df.shift(1))


def normalize_for_comparison(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with any NaNs (common dates), then rebase each column to 100
    at the first remaining date. Useful for "start=100" relative performance.
    """
    df = prices_df.dropna().copy()
    if df.empty:
        return df
    return df / df.iloc[0] * 100.0


def rebase_to_index(series_or_df: pd.Series | pd.DataFrame, base: float = 1.0):
    """
    Rebase a Series/DataFrame to a chosen base at the first row (default 1.0).

    Examples:
        rebase_to_index(price_series, base=1.0)   -> index starting at 1.0
        rebase_to_index(price_df, base=100.0)     -> table starting at 100
    """
    out = series_or_df.copy()
    first = out.iloc[0]
    return out / first * base


# --------------------------- Indicators / Risk --------------------------------

def moving_averages(price_series: pd.Series, windows: Iterable[int]) -> pd.DataFrame:
    """
    Return a DataFrame with one column per moving-average window (MA{w}).
    Example columns: MA20, MA50, MA200.
    """
    out = pd.DataFrame(index=price_series.index)
    for w in windows:
        out[f"MA{w}"] = price_series.rolling(int(w)).mean()
    return out


def rolling_volatility(
    returns: pd.Series,
    window: int,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Rolling standard deviation of returns over `window`. If `annualize`,
    multiply by sqrt(periods_per_year). Drops initial NaNs naturally.
    """
    vol = returns.rolling(int(window)).std()
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    return vol.rename("Rolling volatility" + (" (ann.)" if annualize else ""))


def compute_drawdown(price_series: pd.Series) -> tuple[pd.Series, float]:
    """
    Compute drawdown series and max drawdown from a price series.

    Steps:
    - Build index starting at 1.0
    - Track running peak
    - Drawdown = idx / peak - 1  (0 at peaks, negative otherwise)

    Returns
    -------
    drawdown : pd.Series (<=0)
    max_dd   : float (most negative value)
    """
    idx = rebase_to_index(price_series, base=1.0)
    peak = idx.cummax()
    dd = (idx / peak) - 1.0
    dd.name = "Drawdown"
    return dd, float(dd.min())
