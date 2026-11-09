"""Performance metrics for monthly return series and weight histories."""

import numpy as np
import pandas as pd
from scipy import stats

from hkdiv.backtest import BacktestResult


def ann_mean(r: pd.Series) -> float:
    """Annualised arithmetic mean return (12 x monthly mean)."""
    return 12 * r.mean()


def bankrupt(r: pd.Series) -> bool:
    """True if wealth is wiped out (some monthly return <= -100 %).

    Possible for leveraged long-short portfolios; compounding after that is meaningless.
    """
    return bool((r <= -1).any())


def cagr(r: pd.Series) -> float:
    """Compound annual growth rate (NaN if the strategy went bankrupt)."""
    if bankrupt(r):
        return np.nan
    return (1 + r).prod() ** (12 / len(r)) - 1


def ann_vol(r: pd.Series) -> float:
    return r.std(ddof=1) * np.sqrt(12)


def sharpe(r: pd.Series, rf: pd.Series) -> float:
    """Annualised Sharpe ratio of excess returns r - rf."""
    ex = (r - rf.reindex(r.index)).dropna()
    sd = ex.std(ddof=1)
    return np.nan if sd == 0 else ex.mean() / sd * np.sqrt(12)


def max_drawdown(r: pd.Series) -> float:
    """Largest peak-to-trough fall of cumulative wealth (negative number).

    Wealth starts at 1 before the first return, so an initial loss counts.
    A bankrupt strategy has drawdown -1 (total loss).
    """
    if bankrupt(r):
        return -1.0
    wealth = np.concatenate([[1.0], (1 + r).cumprod().to_numpy()])
    peak = np.maximum.accumulate(wealth)
    return float((wealth / peak - 1).min())


def turnover(res: BacktestResult) -> pd.Series:
    """Monthly turnover Σ_j |w_{j,t} - w_{j,t-1}^+| (DeMiguel et al. 2009).

    w^+ are last month's weights after drifting with realised returns over the holding
    month. The first decision (from cash) is not counted. After a bankruptcy (portfolio
    return <= -100 %) drifted weights are undefined, so turnover is NaN from then on.
    """
    w = res.weights
    r = res.asset_returns
    gross = w * (1 + r)
    drifted = gross.div(gross.sum(axis=1), axis=0)
    # drifted weights of decision t-1 are compared with new weights at t
    to = (w - drifted.shift(1)).abs().sum(axis=1)
    bust = res.returns.index[res.returns <= -1]
    if len(bust):
        to[to.index >= bust[0]] = np.nan
    return to.iloc[1:]


def effective_n(weights: pd.DataFrame) -> pd.Series:
    """Effective number of stocks 1 / Σ w², per decision date."""
    return 1 / (weights**2).sum(axis=1)


def sharpe_test(r1: pd.Series, r2: pd.Series, rf: pd.Series) -> float:
    """Two-sided p-value for H0: equal Sharpe ratios (Jobson-Korkie with Memmel 2003 correction).

    Uses excess returns over rf on the common dates.
    """
    df = pd.concat([r1, r2], axis=1).sub(rf, axis=0).dropna()
    x1, x2 = df.iloc[:, 0], df.iloc[:, 1]
    n = len(df)
    m1, m2 = x1.mean(), x2.mean()
    s1, s2 = x1.std(ddof=1), x2.std(ddof=1)
    s12 = np.cov(x1, x2, ddof=1)[0, 1]
    var = (2 * s1**2 * s2**2 - 2 * s1 * s2 * s12
           + 0.5 * m1**2 * s2**2 + 0.5 * m2**2 * s1**2
           - m1 * m2 / (s1 * s2) * s12**2) / n
    z = (s2 * m1 - s1 * m2) / np.sqrt(var)
    return float(2 * stats.norm.sf(abs(z)))


def summarize(r: pd.Series, rf: pd.Series, res: BacktestResult | None = None,
              baseline: pd.Series | None = None) -> dict:
    """One row of the results table."""
    row = {
        "ann_mean": ann_mean(r),
        "cagr": cagr(r),
        "ann_vol": ann_vol(r),
        "sharpe": sharpe(r, rf),
        "max_drawdown": max_drawdown(r),
        "turnover": np.nan,
        "eff_n": np.nan,
        "n_assets": np.nan,
        "p_sharpe_vs_1N": np.nan,
        "bankrupt": bankrupt(r),
    }
    if res is not None:
        row["turnover"] = turnover(res).mean()
        row["eff_n"] = effective_n(res.weights).mean()
        row["n_assets"] = res.n_assets.mean()
    if baseline is not None and not r.equals(baseline):
        row["p_sharpe_vs_1N"] = sharpe_test(r, baseline, rf)
    return row
