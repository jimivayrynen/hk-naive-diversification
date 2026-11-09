"""Sensitivity analysis: estimation window length and universe size."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from hkdiv import metrics
from hkdiv.backtest import run_backtest
from hkdiv.strategies import Strategy
from hkdiv.universe import UniverseFn, full_history


def subset_universe(subset: Iterable[str]) -> UniverseFn:
    """Eligible stocks (full_history rule) restricted to a fixed subset of tickers."""
    keep = pd.Index(list(subset))

    def universe(returns: pd.DataFrame, t: pd.Timestamp, window: int) -> pd.Index:
        return full_history(returns, t, window).intersection(keep, sort=False)

    return universe


def first_decision(returns: pd.DataFrame, window: int, start: str) -> pd.Timestamp:
    """Earliest decision date >= start with a full window of history."""
    return max(pd.Timestamp(start) + pd.offsets.MonthEnd(0), returns.index[window - 1])


def window_grid(returns: pd.DataFrame, rf: pd.Series, strategies: dict[str, Strategy],
                windows: Iterable[int], start: str) -> pd.DataFrame:
    """Sharpe ratio per (strategy, window), all on the same out-of-sample period.

    The common period starts at the first date where the longest window is available.
    """
    windows = list(windows)
    t0 = max(first_decision(returns, w, start) for w in windows)
    rows = {}
    for w in windows:
        for name, strat in strategies.items():
            res = run_backtest(returns, rf, strat, window=w, start=str(t0.date()))
            rows[(name, w)] = {"sharpe": metrics.sharpe(res.returns, rf),
                               "ann_vol": metrics.ann_vol(res.returns),
                               "bankrupt": metrics.bankrupt(res.returns),
                               "n_assets": res.n_assets.mean(),
                               "first_return": res.returns.index[0].date()}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.names = ["strategy", "window"]
    return out


def random_n_grid(returns: pd.DataFrame, rf: pd.Series, strategies: dict[str, Strategy],
                  sizes: Iterable[int], reps: int, window: int, start: str,
                  seed: int = 0) -> pd.DataFrame:
    """Sharpe ratio per (N, repetition, strategy) for random fixed universes of N stocks.

    Each repetition draws N tickers once from the stocks eligible at the first decision
    date and keeps that set for the whole backtest (members that delist drop out).
    """
    rng = np.random.default_rng(seed)
    t0 = first_decision(returns, window, start)
    pool = full_history(returns, t0, window)
    rows = []
    for n in sizes:
        for rep in range(reps):
            subset = rng.choice(pool, size=n, replace=False)
            uni = subset_universe(subset)
            for name, strat in strategies.items():
                res = run_backtest(returns, rf, strat, window=window, start=str(t0.date()),
                                   universe=uni)
                rows.append({"N": n, "rep": rep, "strategy": name,
                             "sharpe": metrics.sharpe(res.returns, rf),
                             "bankrupt": metrics.bankrupt(res.returns)})
    return pd.DataFrame(rows)


def summarize_random_n(grid: pd.DataFrame, baseline: str = "1/N") -> pd.DataFrame:
    """Mean / median Sharpe per (N, strategy) and the share of repetitions beating 1/N."""
    wide = grid.pivot_table(index=["N", "rep"], columns="strategy", values="sharpe")
    beats = wide.drop(columns=baseline).gt(wide[baseline], axis=0)
    long = wide.stack().rename("sharpe").reset_index()
    out = long.groupby(["N", "strategy"])["sharpe"].agg(["mean", "median", "std"])
    out["share_beats_1N"] = beats.groupby("N").mean().stack()
    out["bankrupt_reps"] = grid.groupby(["N", "strategy"])["bankrupt"].sum()
    return out
