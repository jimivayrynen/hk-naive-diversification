import numpy as np
import pandas as pd
import pytest

from hkdiv import strategies as st

# IEDA 3330 Homework 2, Problem 3
MU = np.array([1.0, 2.0, 4.0])
SIGMA = np.array([[2.0, 2.0, 8.0], [2.0, 4.0, 8.0], [8.0, 8.0, 34.0]])


def test_gmvp_closed_matches_homework2_problem3():
    # Σ⁻¹e = (13/2, 0, -3/2), eᵀΣ⁻¹e = 5  ->  w = (1.3, 0, -0.3)
    assert np.linalg.solve(SIGMA, np.ones(3)) == pytest.approx([6.5, 0.0, -1.5])
    w, info = st.gmvp_closed(SIGMA)
    assert w == pytest.approx([1.3, 0.0, -0.3])
    assert w @ MU == pytest.approx(0.1)  # GMVP mean
    assert w @ SIGMA @ w == pytest.approx(0.2)  # GMVP variance = 1 / eᵀΣ⁻¹e
    assert not info["singular"]


def test_tangency_closed_homework2_rf0():
    # Σ⁻¹r̄ = (0, 1/2, 0)  ->  tangency with rf = 0 is (0, 1, 0)
    w, info = st.tangency_closed(MU, SIGMA, 0.0)
    assert w == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    assert not info["fallback"]


def test_tangency_closed_falls_back_when_rf_above_gmvp_mean():
    # GMVP mean is 0.1; with rf = 1 the denominator eᵀΣ⁻¹(r̄ - rf·e) = 1/2 - 5 < 0
    w, info = st.tangency_closed(MU, SIGMA, 1.0)
    assert info["fallback"]
    assert w == pytest.approx([1.3, 0.0, -0.3])


def test_tangency_closed_falls_back_when_denominator_near_zero():
    # GMVP mean is 0.1: rf just below it -> denominator > 0 but tiny -> gross exposure explodes
    w_raw = np.linalg.solve(SIGMA, MU - 0.0999) / np.linalg.solve(SIGMA, MU - 0.0999).sum()
    assert np.abs(w_raw).sum() > st.MAX_GROSS
    w, info = st.tangency_closed(MU, SIGMA, 0.0999)
    assert info["fallback"]
    assert w == pytest.approx([1.3, 0.0, -0.3])


def test_tangency_long_only_matches_closed_form_when_interior():
    mu = np.array([0.01, 0.02, 0.03])
    cov = np.diag([0.01, 0.02, 0.04])  # closed-form tangency is all positive here
    w_cf, _ = st.tangency_closed(mu, cov, 0.0)
    assert (w_cf > 0).all()
    w_lo, info = st.tangency_long_only(mu, cov, 0.0)
    assert not info["fallback"]
    assert w_lo == pytest.approx(w_cf, abs=1e-6)


def test_tangency_long_only_beats_every_single_stock_sharpe():
    rng = np.random.default_rng(5)
    x = rng.normal(0.01, 0.05, (60, 6))
    mu, cov = x.mean(axis=0), np.cov(x, rowvar=False)
    w, _ = st.tangency_long_only(mu, cov, 0.0)
    best = (w @ mu) / np.sqrt(w @ cov @ w)
    singles = mu / np.sqrt(np.diag(cov))
    assert best >= singles.max() - 1e-9


def test_singular_sigma_uses_pseudo_inverse():
    rng = np.random.default_rng(1)
    window = pd.DataFrame(rng.normal(0, 0.05, (10, 20)))  # N > T -> singular
    res = st.STRATEGIES["GMVP"](window, 0.0)
    assert res.info["singular"]
    assert res.weights.sum() == pytest.approx(1.0)


def test_long_only_gmvp_equals_closed_form_when_interior():
    cov = np.diag([0.01, 0.02, 0.04])
    w_lo, _ = st.gmvp_long_only(cov)
    w_cf, _ = st.gmvp_closed(cov)
    assert w_lo == pytest.approx(w_cf, abs=1e-6)


def test_long_only_gmvp_homework2_has_no_short():
    w, _ = st.gmvp_long_only(SIGMA)
    assert (w >= 0).all()
    assert w.sum() == pytest.approx(1.0)
    # best long-only portfolio: stock 1 only (stock 3 was shorted in the closed form)
    assert w == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)


def test_tangency_long_only_fallback_when_no_positive_excess():
    w, info = st.tangency_long_only(MU, SIGMA, 10.0)
    assert info["fallback"]
    assert w == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)


def test_ledoit_wolf_diagnostics():
    rng = np.random.default_rng(2)
    window = pd.DataFrame(rng.normal(0, 0.05, (60, 30)))
    _, _, info = st.ledoit_wolf_moments(window)
    assert 0 <= info["lw_delta"] <= 1
    assert info["cond_lw"] < info["cond_sample"]


@pytest.mark.parametrize("name", list(st.STRATEGIES))
def test_all_strategies_weights_sum_to_one_and_long_only_nonnegative(name, random_returns):
    window = random_returns.iloc[:30, :7]
    res = st.STRATEGIES[name](window, 0.001)
    assert res.weights.index.equals(window.columns)
    assert res.weights.sum() == pytest.approx(1.0)
    if name in st.LONG_ONLY:
        assert (res.weights >= 0).all()
