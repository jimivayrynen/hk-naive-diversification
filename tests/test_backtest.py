import numpy as np
import pandas as pd
import pytest

from hkdiv import strategies as st
from hkdiv.backtest import assert_no_lookahead, run_backtest
from tests.conftest import month_ends


def test_equal_weight_constant_returns_is_two_percent():
    idx = month_ends(24)
    rets = pd.DataFrame({"A": 0.01, "B": 0.03}, index=idx)
    rf = pd.Series(0.0, index=idx)
    res = run_backtest(rets, rf, st.equal_weight, window=12, start=str(idx[11].date()))
    assert len(res.returns) == 12
    assert res.returns.to_numpy() == pytest.approx(0.02)


def test_returns_indexed_by_holding_month():
    idx = month_ends(24)
    rets = pd.DataFrame({"A": np.arange(24) / 100}, index=idx)
    rf = pd.Series(0.0, index=idx)
    res = run_backtest(rets, rf, st.equal_weight, window=12, start=str(idx[11].date()))
    assert res.returns.index[0] == idx[12]
    assert res.weights.index[0] == idx[11]
    assert res.returns.iloc[0] == pytest.approx(0.12)  # return of month 13, not 12


def test_stock_ending_midway_is_dropped_and_loop_survives(random_returns, rf_const):
    # S7 has no returns from month 31 (index 30) on
    res = run_backtest(random_returns, rf_const, st.equal_weight, window=12,
                       start=str(random_returns.index[20].date()))
    last_hold_s7 = random_returns.index[29]  # last decision with S7's window complete
    assert res.weights.loc[last_hold_s7, "S7"] > 0
    assert (res.weights.loc[res.weights.index > last_hold_s7, "S7"] == 0).all()
    assert res.n_assets.loc[last_hold_s7] == 8
    assert res.n_assets.iloc[-1] == 7


def test_delisting_during_holding_month_earns_zero():
    idx = month_ends(14)
    rets = pd.DataFrame({"A": 0.02, "B": 0.04}, index=idx)
    rets.loc[idx[12]:, "B"] = np.nan  # B delists in the first holding month
    rf = pd.Series(0.0, index=idx)
    res = run_backtest(rets, rf, st.equal_weight, window=12, start=str(idx[11].date()))
    assert res.returns.iloc[0] == pytest.approx(0.5 * 0.02 + 0.5 * 0.0)


def test_lookahead_assert_fires():
    idx = month_ends(5)
    with pytest.raises(AssertionError, match="look-ahead"):
        assert_no_lookahead(idx[:4], decision=idx[2], return_date=idx[3])
    with pytest.raises(AssertionError, match="look-ahead"):
        assert_no_lookahead(idx[:3], decision=idx[1], return_date=idx[2])
    assert_no_lookahead(idx[:3], decision=idx[2], return_date=idx[3])  # ok


@pytest.mark.parametrize("name", list(st.STRATEGIES))
def test_every_weight_vector_sums_to_one(name, random_returns, rf_const):
    res = run_backtest(random_returns, rf_const, st.STRATEGIES[name], window=24,
                       start=str(random_returns.index[23].date()))
    assert res.weights.sum(axis=1).to_numpy() == pytest.approx(1.0)
    if name in st.LONG_ONLY:
        assert (res.weights.to_numpy() >= 0).all()


def test_universe_cannot_see_the_future(random_returns, rf_const):
    def peeking(returns, t, window):  # survivorship look-ahead: needs next month's return
        t_next = returns.index[returns.index.get_loc(t) + 1]
        return returns.columns[returns.loc[t_next].notna()]

    with pytest.raises((KeyError, IndexError)):
        run_backtest(random_returns, rf_const, st.equal_weight, window=12,
                     start=str(random_returns.index[20].date()), universe=peeking)
