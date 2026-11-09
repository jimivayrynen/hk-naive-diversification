"""Run all strategies out-of-sample and write reports/results.csv and reports/figures/."""
import pandas as pd

from hkdiv import data, metrics, plots
from hkdiv.backtest import run_backtest
from hkdiv.strategies import STRATEGIES

WINDOW = 60
START = "2014-01"  # first decision; first return 2014-02

REPORTS = data.ROOT / "reports"
FIGS = REPORTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

returns = data.load_returns()
bench = data.load_bench()
rf = bench["RF"]

results = {}
for name, strategy in STRATEGIES.items():
    print(f"running {name} ...")
    results[name] = run_backtest(returns, rf, strategy, window=WINDOW, start=START)

oos = {k: r.returns for k, r in results.items()}
period = oos["1/N"].index
hsi = bench["HSI"].reindex(period)
if hsi.isna().any():
    raise ValueError("HSI has missing values in the backtest period")
oos["HSI"] = hsi

rows = {k: metrics.summarize(r.returns, rf, r, baseline=oos["1/N"]) for k, r in results.items()}
rows["HSI"] = metrics.summarize(hsi, rf, baseline=oos["1/N"])
table = pd.DataFrame.from_dict(rows, orient="index")
table.index.name = "strategy"
table.to_csv(REPORTS / "results.csv", float_format="%.4f")

print(f"\nOut-of-sample {period[0].date()} - {period[-1].date()} ({len(period)} months), "
      f"window {WINDOW}")
print(table.round(3).to_string())

fallbacks = {k: int(r.info["fallback"].sum()) for k, r in results.items()
             if "fallback" in r.info}
singular = {k: int(r.info["singular"].sum()) for k, r in results.items()
            if "singular" in r.info}
print("\ntangency fallbacks to GMVP (months):", fallbacks)
print("months with singular Σ (pseudo-inverse used):", singular)

# Figures
plots.cumulative(oos, plots.LONG_ONLY_SET, "Cumulative return, long-only strategies",
                 FIGS / "cumulative_long_only.png")
plots.cumulative(oos, plots.UNCONSTRAINED_SET, "Cumulative return, unconstrained strategies",
                 FIGS / "cumulative_unconstrained.png", log=True)
plots.single(results["1/N"].n_assets, "Number of stocks in the universe", "Stocks",
             FIGS / "n_assets.png")
plots.small_multiples({k: metrics.turnover(r) for k, r in results.items()},
                      "Monthly turnover", "Σ|Δw| (log scale)", FIGS / "turnover.png", log=True)
plots.small_multiples({k: metrics.effective_n(r.weights) for k, r in results.items()},
                      "Effective number of stocks", "1 / Σw² (log scale)",
                      FIGS / "effective_n.png", log=True, sharey=False)
plots.lw_diagnostics(results["GMVP-LW"].info, FIGS / "lw_diagnostics.png")
print("\nwrote", REPORTS / "results.csv", "and figures to", FIGS)
