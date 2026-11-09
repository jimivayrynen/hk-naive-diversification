"""Portfolio strategies.

Notation follows the course: r̄ = mean return vector, Σ = covariance matrix,
e = vector of ones, rf = risk-free rate for the holding month.

Every strategy has the signature `strategy(window, rf) -> StrategyResult`, where
`window` is a (months x stocks) DataFrame of past returns without missing values.
Weights are returned as a Series indexed by the window's tickers and sum to one.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

# Σ is treated as singular above this condition number and inverted with the
# Moore-Penrose pseudo-inverse instead (happens with the sample Σ whenever N >= T).
SINGULAR_COND = 1e12

# Tangency denominator eᵀΣ⁻¹(r̄ - rf·e) counts as "≈ 0" when the normalised portfolio would
# have gross exposure Σ|w| above this (i.e. denominator < Σ|x| / MAX_GROSS).
MAX_GROSS = 100.0


@dataclass
class StrategyResult:
    weights: pd.Series
    info: dict = field(default_factory=dict)


Strategy = Callable[[pd.DataFrame, float], StrategyResult]


# --- Estimators -----------------------------------------------------------------


def sample_moments(window: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sample r̄ and Σ (ddof=1)."""
    x = window.to_numpy()
    cov = np.cov(x, rowvar=False, ddof=1)
    return x.mean(axis=0), cov, {"cond_sample": np.linalg.cond(cov)}


def ledoit_wolf_moments(window: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sample r̄ and Ledoit-Wolf shrunk Σ = (1-δ)S + δ·μI (sklearn, scaled-identity target).

    info contains δ and the condition numbers of the sample and shrunk Σ.
    """
    x = window.to_numpy()
    lw = LedoitWolf().fit(x)
    sample = np.cov(x, rowvar=False, ddof=1)
    info = {
        "lw_delta": float(lw.shrinkage_),
        "cond_sample": np.linalg.cond(sample),
        "cond_lw": np.linalg.cond(lw.covariance_),
    }
    return x.mean(axis=0), lw.covariance_, info


def _solve(cov: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, bool]:
    """Σ⁻¹b, using the pseudo-inverse if Σ is (numerically) singular."""
    if np.linalg.cond(cov) > SINGULAR_COND:
        return np.linalg.pinv(cov, rcond=1e-10, hermitian=True) @ b, True
    return np.linalg.solve(cov, b), False


# --- Weight rules ---------------------------------------------------------------


def gmvp_closed(cov: np.ndarray) -> tuple[np.ndarray, dict]:
    """GMVP: w = Σ⁻¹e / (eᵀΣ⁻¹e)."""
    e = np.ones(cov.shape[0])
    x, singular = _solve(cov, e)
    return x / x.sum(), {"singular": singular}


def tangency_closed(mu: np.ndarray, cov: np.ndarray, rf: float) -> tuple[np.ndarray, dict]:
    """Tangency portfolio: w = Σ⁻¹(r̄ - rf·e) / (eᵀΣ⁻¹(r̄ - rf·e)).

    The denominator equals (eᵀΣ⁻¹e)(GMVP mean - rf). If it is <= 0 (rf above the GMVP mean)
    the formula lands on the inefficient branch; if it is ≈ 0 the weights explode. In both
    cases (denominator <= Σ|x| / MAX_GROSS, i.e. gross exposure > MAX_GROSS) fall back to
    the GMVP and flag it.
    """
    x, singular = _solve(cov, mu - rf)
    denom = x.sum()
    if denom <= np.abs(x).sum() / MAX_GROSS:
        w, _ = gmvp_closed(cov)
        return w, {"singular": singular, "fallback": True}
    return x / denom, {"singular": singular, "fallback": False}


def _psd(cov: np.ndarray) -> np.ndarray:
    """Symmetrised Σ scaled to unit average variance (same argmin, better solver accuracy)."""
    cov = (cov + cov.T) / 2
    return cov / np.mean(np.diag(cov))


def _solve_qp(prob: cp.Problem, var: cp.Variable, name: str) -> np.ndarray:
    """OSQP with solution polishing (exact zeros at the w >= 0 bounds); Clarabel as backup."""
    try:
        prob.solve(solver=cp.OSQP, eps_abs=1e-10, eps_rel=1e-10, polishing=True,
                   max_iter=200_000)
    except cp.SolverError:
        pass
    if prob.status != cp.OPTIMAL:
        prob.solve(solver=cp.CLARABEL)
    if var.value is None or prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"{name} failed: {prob.status}")
    return var.value


def _clean(w: np.ndarray) -> np.ndarray:
    """Remove solver noise from long-only weights and renormalise."""
    w = np.where(w < 1e-8, 0.0, w)
    return w / w.sum()


def gmvp_long_only(cov: np.ndarray) -> tuple[np.ndarray, dict]:
    """min wᵀΣw  s.t.  eᵀw = 1, w >= 0."""
    n = cov.shape[0]
    w = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(_psd(cov)))),
                      [cp.sum(w) == 1, w >= 0])
    return _clean(_solve_qp(prob, w, "GMVP long-only")), {}


def tangency_long_only(mu: np.ndarray, cov: np.ndarray, rf: float) -> tuple[np.ndarray, dict]:
    """Long-only maximum Sharpe ratio portfolio.

    Solved as the convex problem  min yᵀΣy  s.t. (r̄ - rf·e)ᵀy = 1, y >= 0, w = y / eᵀy.
    If no stock has r̄ > rf the problem is infeasible: fall back to GMVP long-only.
    """
    excess = mu - rf
    if excess.max() <= 0:
        w, _ = gmvp_long_only(cov)
        return w, {"fallback": True}
    y = cp.Variable(len(mu))
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(_psd(cov)))),
                      [excess @ y == 1, y >= 0])
    return _clean(_solve_qp(prob, y, "tangency long-only")), {"fallback": False}


# --- Strategies -----------------------------------------------------------------


def equal_weight(window: pd.DataFrame, rf: float) -> StrategyResult:
    """Naive 1/N."""
    n = window.shape[1]
    return StrategyResult(pd.Series(1.0 / n, index=window.columns))


def _make(estimator, rule, uses_mu: bool) -> Strategy:
    def strategy(window: pd.DataFrame, rf: float) -> StrategyResult:
        mu, cov, info = estimator(window)
        w, rule_info = rule(mu, cov, rf) if uses_mu else rule(cov)
        return StrategyResult(pd.Series(w, index=window.columns), {**info, **rule_info})

    return strategy


STRATEGIES: dict[str, Strategy] = {
    "1/N": equal_weight,
    "GMVP": _make(sample_moments, gmvp_closed, uses_mu=False),
    "Tangent": _make(sample_moments, tangency_closed, uses_mu=True),
    "GMVP-LO": _make(sample_moments, gmvp_long_only, uses_mu=False),
    "Tangent-LO": _make(sample_moments, tangency_long_only, uses_mu=True),
    "GMVP-LW": _make(ledoit_wolf_moments, gmvp_closed, uses_mu=False),
    "Tangent-LW": _make(ledoit_wolf_moments, tangency_closed, uses_mu=True),
    "GMVP-LW-LO": _make(ledoit_wolf_moments, gmvp_long_only, uses_mu=False),
    "Tangent-LW-LO": _make(ledoit_wolf_moments, tangency_long_only, uses_mu=True),
}

LONG_ONLY = {"1/N", "GMVP-LO", "Tangent-LO", "GMVP-LW-LO", "Tangent-LW-LO"}
