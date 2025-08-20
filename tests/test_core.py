# tests/test_core.py

"""
Unit tests for pure helpers in core.py.

Covers:
    - load_prices: shape normalization (Series→DataFrame), MultiIndex handling, error cases
    - log_returns: basic log-return math and shape preservation
    - normalize_for_comparison: common-date alignment + start=100 rebasing
    - moving_averages: rolling mean columns (MA{w})
    - rolling_volatility: windowed std of returns (annualized)
    - compute_drawdown: drawdown series and max drawdown
    - rebase_to_index: rebasing a series/table to a chosen base (1.0 or 100)

How to run
    # Option A (recommended): 
    make test run

    # Option B:
    PYTHONPATH=. pytest -q
"""

from datetime import date
import numpy as np
import pandas as pd

from core import (
    load_prices,
    log_returns,
    normalize_for_comparison,
    moving_averages,
    rolling_volatility,
    compute_drawdown,
    rebase_to_index,
)

# -----------------------------------------------------------------------------
# Fixtures (helpers)
# -----------------------------------------------------------------------------
def _bday_index(n: int = 5, start: str = "2024-01-01") -> pd.DatetimeIndex:
    """Business-day date index for small toy datasets."""
    return pd.bdate_range(start=start, periods=n)


def _fake_download_single(**kwargs) -> pd.DataFrame:
    """
    Fake yfinance.download(...) for a single ticker.

    Returns a DataFrame where "Close" is a plain column (i.e., selecting
    df["Close"] yields a Series in core.load_prices).
    """
    idx = _bday_index(3)
    return pd.DataFrame({"Close": [100.0, 105.0, 110.0]}, index=idx)


def _fake_download_multi(**kwargs) -> pd.DataFrame:
    """
    Fake yfinance.download(...) for multiple tickers.

    Returns a DataFrame with a MultiIndex on columns, level 0 = "Close",
    level 1 = ticker symbol — matching yfinance's typical shape.
    """
    idx = _bday_index(3)
    arrays = [["Close", "Close"], ["AAA", "BBB"]]
    cols = pd.MultiIndex.from_arrays(arrays)
    data = np.column_stack([[10.0, 11.0, 12.0], [20.0, 22.0, 24.0]])
    return pd.DataFrame(data=data, index=idx, columns=cols)


def _fake_download_no_close(**kwargs) -> pd.DataFrame:
    """Fake download that lacks a 'Close' key → core.load_prices should return empty."""
    idx = _bday_index(3)
    return pd.DataFrame({"Open": [1, 2, 3]}, index=idx)


# -----------------------------------------------------------------------------
# Tests: log_returns
# -----------------------------------------------------------------------------
def test_log_returns_series_basic():
    """Series input → same shape; first element NaN; values match ln(Pt/Pt-1)."""
    s = pd.Series([100.0, 105.0, 110.25], index=_bday_index(3))
    r = log_returns(s)
    assert r.shape == s.shape
    assert np.isnan(r.iloc[0])
    np.testing.assert_allclose(r.iloc[1], np.log(105 / 100))
    np.testing.assert_allclose(r.iloc[2], np.log(110.25 / 105))


def test_log_returns_dataframe_shape_and_cols():
    """DataFrame input → preserves columns; first row all NaN."""
    df = pd.DataFrame(
        {"AAA": [100.0, 110.0, 121.0], "BBB": [50.0, 55.0, 60.5]},
        index=_bday_index(3),
    )
    r = log_returns(df)
    assert list(r.columns) == ["AAA", "BBB"]
    assert r.isna().iloc[0].all()


# -----------------------------------------------------------------------------
# Tests: normalize_for_comparison
# -----------------------------------------------------------------------------
def test_normalize_for_comparison_basic():
    """
    Rebase each series so the first common row becomes 100,
    and relative scaling holds at later dates.
    """
    df = pd.DataFrame(
        {"AAA": [100.0, 110.0, 120.0], "BBB": [50.0, 55.0, 60.0]},
        index=_bday_index(3),
    )
    norm = normalize_for_comparison(df)
    # First row should be 100 for all tickers
    assert np.allclose(norm.iloc[0].values, [100.0, 100.0])
    # Last row: AAA 120/100*100 = 120; BBB 60/50*100 = 120
    assert np.allclose(norm.iloc[-1].values, [120.0, 120.0])


def test_normalize_for_comparison_handles_nans_by_dropping_rows():
    """Rows with any NaNs are dropped to ensure a shared baseline date."""
    df = pd.DataFrame(
        {"AAA": [100.0, np.nan, 120.0], "BBB": [50.0, 55.0, 60.0]},
        index=_bday_index(3),
    )
    norm = normalize_for_comparison(df)
    # Middle row dropped → resulting index length 2
    assert len(norm) == 2
    # First common row should be scaled to 100
    assert np.allclose(norm.iloc[0].values, [100.0, 100.0])


# -----------------------------------------------------------------------------
# Tests: load_prices (using fake download functions)
# -----------------------------------------------------------------------------
def test_load_prices_no_tickers_returns_empty():
    """No tickers → early return empty DataFrame."""
    out = load_prices([], date(2024, 1, 1), date(2024, 1, 10), download_fn=_fake_download_single)
    assert out.empty


def test_load_prices_single_ticker_becomes_one_column_df():
    """
    Single ticker input where df['Close'] is a Series → ensure core normalizes to a
    one-column DataFrame named after the ticker.
    """
    out = load_prices(["AAA"], date(2024, 1, 1), date(2024, 1, 10), download_fn=_fake_download_single)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["AAA"]
    assert (out["AAA"] > 0).all()


def test_load_prices_multi_ticker_flattens_multiindex():
    """Multi-ticker input → extract the 'Close' level and get columns ['AAA','BBB']."""
    out = load_prices(["AAA", "BBB"], date(2024, 1, 1), date(2024, 1, 10), download_fn=_fake_download_multi)
    assert list(out.columns) == ["AAA", "BBB"]
    assert out.shape[0] == 3  # 3 business days fed in


def test_load_prices_missing_close_returns_empty():
    """If 'Close' is absent in the downloaded table, return an empty DataFrame."""
    out = load_prices(["AAA"], date(2024, 1, 1), date(2024, 1, 10), download_fn=_fake_download_no_close)
    assert out.empty


# -----------------------------------------------------------------------------
# Tests: indicators / risk helpers
# -----------------------------------------------------------------------------
def test_moving_averages():
    """moving_averages should add MA{w} columns with correct rolling means."""
    s = pd.Series([1, 2, 3, 4, 5], index=pd.bdate_range("2024-01-01", periods=5))
    ma = moving_averages(s, [3])
    assert "MA3" in ma.columns
    # Last MA3 = mean of the last 3 values: (3 + 4 + 5) / 3 = 4.0
    assert ma["MA3"].iloc[-1] == 4.0


def test_rolling_volatility_ann():
    """
    rolling_volatility should produce a series with some non-NaN values
    and, when annualized, include 'ann.' in the name.
    """
    r = pd.Series([0.0, 0.01, -0.01, 0.02, -0.02], index=pd.bdate_range("2024-01-01", periods=5))
    vol = rolling_volatility(r, window=3, annualize=True)
    assert vol.notna().sum() >= 1
    assert "ann." in vol.name


def test_compute_drawdown():
    """
    Drawdown at 90 vs prior peak 120 is 90/120 - 1 = -0.25 (i.e., -25%),
    and max drawdown should equal the lowest drawdown in the series.
    """
    s = pd.Series([100, 120, 90, 95, 130], index=pd.bdate_range("2024-01-01", periods=5))
    dd, max_dd = compute_drawdown(s)
    assert np.isclose(dd.iloc[2], -0.25)
    assert np.isclose(max_dd, -0.25)


def test_rebase_to_index():
    """Rebase to 100: first value becomes 100; last value scaled accordingly."""
    s = pd.Series([50, 75, 100], index=pd.bdate_range("2024-01-01", periods=3))
    idx = rebase_to_index(s, base=100)
    assert np.isclose(idx.iloc[0], 100.0)
    assert np.isclose(idx.iloc[-1], 200.0)  # 100/50 * 100
