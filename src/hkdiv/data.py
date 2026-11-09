"""Loading and cleaning of the Bloomberg data.

data/raw/data.csv holds monthly *total return index* levels (dividends reinvested),
one column per ticker. data/raw/BENCH.csv holds HSI price / total return levels and
1M HIBOR in annual percent.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Tickers removed from the universe, with the reason.
EXCLUDE = {
    "13 HK": "reused code: data is HUTCHMED (listed 2021), not Hutchison Whampoa",
    "1880 HK": "reused code: data is China Tourism Group Duty Free (2022), not Belle",
}


def read_levels(path: Path) -> pd.DataFrame:
    """Read a Bloomberg CSV with a Date index and validate the index."""
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    check_index(df)
    return df


def check_index(df: pd.DataFrame) -> None:
    """Raise if the date index is not strictly increasing and unique."""
    if df.index.has_duplicates:
        raise ValueError(f"duplicate dates: {list(df.index[df.index.duplicated()])}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("date index is not increasing")


def clean_levels(
    levels: pd.DataFrame, exclude: dict[str, str] | None = None
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Drop all-empty columns and excluded tickers.

    Returns (cleaned levels, dropped empty columns, dropped excluded columns).
    """
    exclude = EXCLUDE if exclude is None else exclude
    empty = list(levels.columns[levels.isna().all()])
    excluded = [c for c in levels.columns if c in exclude and c not in empty]
    return levels.drop(columns=empty + excluded), empty, excluded


def to_returns(levels: pd.DataFrame) -> pd.DataFrame:
    """Monthly simple returns from index levels.

    No forward-filling: a stock has returns only between its first and last level,
    so late listings and delistings show up as NaN. The first row is dropped.
    """
    return levels.pct_change(fill_method=None).iloc[1:]


# Runs of at least this many consecutive exactly-zero monthly returns are treated as
# trading suspensions (Bloomberg carries the last price forward).
SUSPENSION_RUN = 3


def mask_suspensions(returns: pd.DataFrame, min_run: int = SUSPENSION_RUN
                     ) -> tuple[pd.DataFrame, list[tuple[str, pd.Timestamp, pd.Timestamp, int]]]:
    """Set runs of >= min_run consecutive zero returns to NaN.

    A suspended stock cannot be traded and its stale zero returns would understate its
    variance (GMVP-type strategies would overweight it). As NaN, the stock drops out of
    the universe while the suspension is inside the estimation window, and earns 0 (cash)
    if a suspension starts while it is held.

    Returns (masked returns, list of (ticker, first, last, length) runs).
    """
    out = returns.copy()
    runs = []
    for col in returns.columns:
        zero = returns[col].eq(0)
        run_id = zero.ne(zero.shift()).cumsum()
        for _, block in zero.groupby(run_id):
            if block.iloc[0] and len(block) >= min_run:
                out.loc[block.index, col] = np.nan
                runs.append((col, block.index[0], block.index[-1], len(block)))
    return out, runs


def bench_returns(bench: pd.DataFrame) -> pd.DataFrame:
    """HSI total return and monthly risk-free rate.

    HSI: months where HSI_TotalReturn equals HSI_Price (2009-01 - 2013-01 in the
    Bloomberg series, i.e. no dividends) are set to NaN, so the benchmark is never
    compared on a price-only basis against total-return stock data.

    RF: 1M HIBOR observed at the end of month t-1 is the rate earned during month t,
    converted from annual percent to a monthly decimal.
    """
    out = pd.DataFrame(
        {
            "HSI": bench["HSI_TotalReturn"].pct_change(fill_method=None),
            "RF": bench["HIBOR_1M"].shift(1) / 100 / 12,
        }
    )
    price_only = np.isclose(bench["HSI_TotalReturn"], bench["HSI_Price"])
    out.loc[price_only, "HSI"] = np.nan
    return out.iloc[1:]


def extreme_returns(returns: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """All (date, ticker) returns with |r| > threshold, sorted."""
    s = returns.stack()
    return s[s.abs() > threshold].sort_values()


def load_returns() -> pd.DataFrame:
    """Cleaned stock returns from data/processed (run scripts/01_clean_data.py first)."""
    return read_levels(PROCESSED / "returns.csv")


def load_bench() -> pd.DataFrame:
    """HSI and RF from data/processed (run scripts/01_clean_data.py first)."""
    return read_levels(PROCESSED / "bench.csv")
