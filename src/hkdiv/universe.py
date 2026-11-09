"""Investable universe at each decision date.

Temporary rule until historical HSI membership data is available: a stock is
eligible at decision date t if it has no missing return in the estimation window
t-window+1 .. t. Uses only information available at t (whether the stock survives
month t+1 is *not* checked, to avoid look-ahead).
"""

from collections.abc import Callable, Iterable

import pandas as pd

from hkdiv.data import EXCLUDE

UniverseFn = Callable[[pd.DataFrame, pd.Timestamp, int], pd.Index]


def window_slice(returns: pd.DataFrame, t: pd.Timestamp, window: int) -> pd.DataFrame:
    """The `window` rows of returns ending at (and including) decision date t."""
    pos = returns.index.get_loc(t)
    if pos + 1 < window:
        raise ValueError(f"not enough history at {t.date()}: {pos + 1} < {window} months")
    return returns.iloc[pos + 1 - window : pos + 1]



def full_history(
    returns: pd.DataFrame, t: pd.Timestamp, window: int, exclude: Iterable[str] = tuple(EXCLUDE)
) -> pd.Index:
    """Tickers with a complete return history in the window ending at t.

    Tickers in `exclude` (default: hkdiv.data.EXCLUDE, the reused codes) are never eligible,
    even if they were not already dropped during cleaning.
    """
    win = window_slice(returns, t, window)
    ok = win.columns[win.notna().all()]
    return ok.difference(pd.Index(list(exclude)), sort=False)
