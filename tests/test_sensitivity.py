import pandas as pd
import pytest

from hkdiv import sensitivity as sens
from hkdiv import strategies as st


def test_subset_universe_restricts_eligible(random_returns):
    uni = sens.subset_universe(["S1", "S7", "NOT_THERE"])
    t = random_returns.index[35]  # S7 has delisted by then
    assert list(uni(random_returns, t, 12)) == ["S1"]


def test_first_decision_respects_history(random_returns):
    assert sens.first_decision(random_returns, 12, "2000-01") == random_returns.index[11]
    late = random_returns.index[20]
    assert sens.first_decision(random_returns, 12, str(late.date())) == late


def test_window_grid_uses_common_period(random_returns, rf_const):
    strategies = {"1/N": st.equal_weight}
    out = sens.window_grid(random_returns, rf_const, strategies, [6, 12], start="2000-01")
    firsts = out["first_return"].unique()
    assert len(firsts) == 1
    assert firsts[0] == random_returns.index[12].date()


def test_random_n_grid_and_summary(random_returns, rf_const):
    strategies = {"1/N": st.equal_weight, "GMVP-LO": st.STRATEGIES["GMVP-LO"]}
    grid = sens.random_n_grid(random_returns, rf_const, strategies, sizes=[3], reps=4,
                              window=12, start="2000-01")
    assert len(grid) == 8
    summ = sens.summarize_random_n(grid)
    assert summ.loc[(3, "GMVP-LO"), "share_beats_1N"] == pytest.approx(
        grid.pivot_table(index="rep", columns="strategy", values="sharpe")
         .pipe(lambda w: (w["GMVP-LO"] > w["1/N"]).mean())
    )
    assert pd.isna(summ.loc[(3, "1/N"), "share_beats_1N"])
