"""Rolling-window out-of-sample backtest.

At each decision date t (month end):
  1. universe = stocks eligible at t (see hkdiv.universe)
  2. estimate on the window t-window+1 .. t and form weights w_t
  3. hold w_t over month t+1; the portfolio return is recorded at date t+1.

A stock that delists during the holding month has a missing return at t+1; it is
treated as cash earning 0 (its weight is kept, its return set to 0).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hkdiv.strategies import Strategy
from hkdiv.universe import UniverseFn, full_history, window_slice


@dataclass
class BacktestResult:
    returns: pd.Series  # portfolio return, indexed by holding month t+1
    weights: pd.DataFrame  # decision date t x all tickers (0 = not held)
    n_assets: pd.Series  # universe size at t
    info: pd.DataFrame  # strategy diagnostics per decision date t
    asset_returns: pd.DataFrame  # realised stock returns at t+1 (missing -> 0), aligned with weights


def assert_no_lookahead(window_index: pd.DatetimeIndex, decision: pd.Timestamp,
                        return_date: pd.Timestamp) -> None:
    """The estimation window must end exactly at the decision date, before the return date."""
    if window_index.max() > decision or window_index.max() >= return_date:
        raise AssertionError(
            f"look-ahead: window ends {window_index.max().date()}, decision "
            f"{decision.date()}, return {return_date.date()}"
        )
    if decision >= return_date:
        raise AssertionError("decision date must be before the return date")


def run_backtest(
    returns: pd.DataFrame,
    rf: pd.Series,
    strategy: Strategy,
    window: int = 60,
    start: str = "2014-01",
    end: str | None = None,
    universe: UniverseFn = full_history,
) -> BacktestResult:
    """Run `strategy` with monthly rebalancing.

    returns: stock returns (dates x tickers), NaN where not listed.
    rf: monthly risk-free rate indexed like returns; rf at date t+1 is the rate for
        holding month t+1, which is known at t (HIBOR lagged in hkdiv.data).
    start / end: first / last *decision* month (inclusive).
    """
    dates = returns.index
    decisions = dates[(dates >= pd.Timestamp(start)) & (dates < dates[-1])]
    if end is not None:
        decisions = decisions[decisions <= pd.Timestamp(end) + pd.offsets.MonthEnd(0)]

    port, w_rows, r_rows, n_assets, infos = {}, {}, {}, {}, {}
    for t in decisions:
        t_next = dates[dates.get_loc(t) + 1]
        known = returns.loc[:t]  # information set at t: nothing after the decision date
        tickers = universe(known, t, window)
        if len(tickers) == 0:
            raise ValueError(f"empty universe at {t.date()}")
        win = window_slice(known, t, window)[tickers]
        assert_no_lookahead(win.index, t, t_next)
        if win.isna().any().any():
            raise ValueError(f"missing returns in estimation window at {t.date()}")

        res = strategy(win, float(rf.loc[t_next]))
        w = res.weights
        if not w.index.equals(tickers):
            raise ValueError("strategy weights are not aligned with the universe")
        if not np.isclose(w.sum(), 1.0):
            raise ValueError(f"weights sum to {w.sum()} at {t.date()}")

        r_next = returns.loc[t_next, tickers].fillna(0.0)  # delisted in month t+1 -> cash
        port[t_next] = float(w @ r_next)
        w_rows[t], r_rows[t], n_assets[t], infos[t] = w, r_next, len(tickers), res.info

    weights = pd.DataFrame(w_rows).T.fillna(0.0)
    asset_returns = pd.DataFrame(r_rows).T.reindex(columns=weights.columns).fillna(0.0)
    return BacktestResult(
        returns=pd.Series(port, name="return"),
        weights=weights,
        n_assets=pd.Series(n_assets, name="n_assets"),
        info=pd.DataFrame(infos).T,
        asset_returns=asset_returns,
    )
