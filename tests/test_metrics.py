import numpy as np
import pandas as pd
import pytest

from hkdiv import metrics as m
from hkdiv import strategies as st
from hkdiv.backtest import run_backtest
from tests.conftest import month_ends


def test_cagr_and_ann_mean():
    r = pd.Series([0.01] * 12)
    assert m.cagr(r) == pytest.approx(1.01**12 - 1)
    assert m.ann_mean(r) == pytest.approx(0.12)


def test_max_drawdown():
    r = pd.Series([0.10, -0.50, 0.20, 0.50])
    # wealth 1, 1.1, 0.55, 0.66, 0.99 -> worst fall 1.1 -> 0.55
    assert m.max_drawdown(r) == pytest.approx(-0.5)
    assert m.max_drawdown(pd.Series([-0.2, 0.1])) == pytest.approx(-0.2)


def test_sharpe():
    idx = month_ends(4)
    r = pd.Series([0.02, 0.00, 0.02, 0.00], index=idx)
    rf = pd.Series(0.005, index=idx)
    assert m.sharpe(r, rf) == pytest.approx(0.005 / r.std() * np.sqrt(12))


def test_effective_n():
    w = pd.DataFrame({"A": [0.25, 1.0], "B": [0.25, 0.0], "C": [0.25, 0.0], "D": [0.25, 0.0]})
    assert m.effective_n(w).tolist() == pytest.approx([4.0, 1.0])


def test_turnover_zero_when_returns_equal_and_known_value_otherwise():
    idx = month_ends(15)
    rf = pd.Series(0.0, index=idx)
    same = pd.DataFrame({"A": 0.01, "B": 0.01}, index=idx)
    res = run_backtest(same, rf, st.equal_weight, window=12, start=str(idx[11].date()))
    assert m.turnover(res).to_numpy() == pytest.approx(0.0)

    diff = pd.DataFrame({"A": 0.0, "B": 1.0}, index=idx)
    res = run_backtest(diff, rf, st.equal_weight, window=12, start=str(idx[11].date()))
    # after a month: A 0.5, B 1.0 -> drifted (1/3, 2/3); back to 1/2 -> |1/6| + |1/6|
    assert m.turnover(res).to_numpy() == pytest.approx(1 / 3)


def test_sharpe_test_known_value():
    # Hand calculation with the Jobson-Korkie/Memmel variance (DeMiguel et al. 2009):
    # x1 = (.02, 0, .04, .02): m1 = .02;  x2 = (.01, .01, .03, -.01): m2 = .01
    # s1² = s2² = 8e-4/3, s12 = 4e-4/3, T = 4
    # T·θ = 2s⁴ - 2s²s12 + ½m1²s² + ½m2²s² - m1m2/s²·s12² = 1.2444e-7
    # z = s(m1 - m2) / sqrt(θ) = 1.6330e-4 / 1.7638e-4 = 0.9258  ->  p = 2(1 - Φ(z)) = 0.3546
    idx = month_ends(4)
    r1 = pd.Series([0.02, 0.0, 0.04, 0.02], index=idx)
    r2 = pd.Series([0.01, 0.01, 0.03, -0.01], index=idx)
    rf = pd.Series(0.0, index=idx)
    assert m.sharpe_test(r1, r2, rf) == pytest.approx(0.3546, abs=1e-3)


def test_sharpe_test_symmetric_and_detects_large_difference():
    rng = np.random.default_rng(3)
    idx = month_ends(240)
    common = rng.normal(0, 0.04, 240)
    r1 = pd.Series(0.015 + common + rng.normal(0, 0.01, 240), index=idx)
    r2 = pd.Series(-0.005 + common + rng.normal(0, 0.01, 240), index=idx)
    rf = pd.Series(0.0, index=idx)
    assert m.sharpe_test(r1, r2, rf) == pytest.approx(m.sharpe_test(r2, r1, rf))
    assert m.sharpe_test(r1, r2, rf) < 0.01


def test_bankrupt_strategy():
    r = pd.Series([0.1, -1.5, 0.2])
    assert m.bankrupt(r)
    assert np.isnan(m.cagr(r))
    assert m.max_drawdown(r) == -1.0
    assert not m.bankrupt(pd.Series([0.1, -0.99]))


def test_turnover_nan_after_bankruptcy():
    idx = month_ends(16)
    rets = pd.DataFrame({"A": 0.01, "B": 0.01}, index=idx)
    rets.loc[idx[13], ["A", "B"]] = -1.2  # wipe-out in the 2nd holding month
    rf = pd.Series(0.0, index=idx)
    res = run_backtest(rets, rf, st.equal_weight, window=3, start=str(idx[11].date()))
    to = m.turnover(res)
    assert not np.isnan(to.loc[idx[12]])
    assert to.loc[idx[13]:].isna().all()
