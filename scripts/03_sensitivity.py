"""Sensitivity to the estimation window (36/60/120) and universe size (N = 10/20/30)."""
import time

from hkdiv import data, sensitivity
from hkdiv.strategies import STRATEGIES

REPORTS = data.ROOT / "reports"
START = "2014-01"
REPS = 20
SEED = 3330

returns = data.load_returns()
rf = data.load_bench()["RF"]

t = time.time()
win = sensitivity.window_grid(returns, rf, STRATEGIES, [36, 60, 120], start=START)
win.to_csv(REPORTS / "sensitivity_window.csv", float_format="%.4f")
print(f"Window length (common period from {win['first_return'].iloc[0]}):")
print(win["sharpe"].unstack("window").round(3).to_string())
print(f"({time.time() - t:.0f} s)\n")

t = time.time()
grid = sensitivity.random_n_grid(returns, rf, STRATEGIES, sizes=[10, 20, 30], reps=REPS,
                                 window=60, start=START, seed=SEED)
summary = sensitivity.summarize_random_n(grid)
summary.to_csv(REPORTS / "sensitivity_n.csv", float_format="%.4f")
print(f"Random universes, {REPS} repetitions per N (seed {SEED}), window 60:")
print(summary.round(3).to_string())
print(f"({time.time() - t:.0f} s)")
