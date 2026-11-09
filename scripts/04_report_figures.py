"""Figures (PDF) and LaTeX tables for the written report -> report/figures, report/tables.

Run after 01_clean_data.py and 03_sensitivity.py (reads reports/sensitivity_*.csv).
Everything shown is derived from the backtest (portfolio statistics), never raw prices.
"""

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hkdiv import data, metrics
from hkdiv.backtest import run_backtest
from hkdiv.strategies import STRATEGIES, sample_moments
from hkdiv.universe import window_slice

WINDOW, START = 60, "2014-01"
OUT = data.ROOT / "report"
FIG, TAB = OUT / "figures", OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
REPORTS = data.ROOT / "reports"
# Separate random generators, so that changing one analysis never changes the numbers of another
RNG_BOOT = np.random.default_rng(3330)  # block bootstrap of Sharpe ratios (Table 2)
RNG_MC = np.random.default_rng(3330)  # Monte Carlo frontier experiment (Figure 1)

# ---------------------------------------------------------------- style
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "lines.linewidth": 1.4,
    }
)
W1, W2 = 6.3, 3.05  # full / half text width (inches)
COL = {
    "1/N": "#1f5aa6",
    "GMVP-LO": "#d9622b",
    "Tangent-LO": "#2a9d6f",
    "GMVP-LW-LO": "#c99a06",
    "Tangent-LW-LO": "#b4457f",
    "HSI": "#555555",
    "GMVP": "#d9622b",
    "Tangent": "#2a9d6f",
    "GMVP-LW": "#c99a06",
    "Tangent-LW": "#b4457f",
}
LO = ["1/N", "GMVP-LO", "Tangent-LO", "GMVP-LW-LO", "Tangent-LW-LO"]
UNC = ["GMVP", "Tangent", "GMVP-LW", "Tangent-LW"]


def write_rows(path, rows):
    """Table body for \\input inside a tabular; ends with \\bottomrule (TeX cannot take a
    \\noalign rule right after an \\input'ed file)."""
    path.write_text("\n".join(rows) + "\n\\bottomrule\n")


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def yearaxis(ax, step=2):
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


# ---------------------------------------------------------------- backtests
returns, bench = data.load_returns(), data.load_bench()
rf = bench["RF"]
res = {k: run_backtest(returns, rf, s, window=WINDOW, start=START) for k, s in STRATEGIES.items()}
oos = {k: r.returns for k, r in res.items()}
period = oos["1/N"].index
oos["HSI"] = bench["HSI"].reindex(period)
rfp = rf.reindex(period)


def ticker_meta():
    """Company name and sector from docs/tickers.md (no price data)."""
    rows = {}
    for line in (data.ROOT / "docs" / "tickers.md").read_text().splitlines():
        m = re.match(r"\| (\d+ HK) \| ([^|]+) \| ([^|]+) \|", line)
        if m:
            rows[m.group(1)] = {"name": m.group(2).strip(), "sector": m.group(3).strip()}
    return pd.DataFrame.from_dict(rows, orient="index")


meta = ticker_meta()


# ---------------------------------------------------------------- extra metrics (report Sec. 2.6)
def ceq(r, gamma=1.0):
    """Certainty-equivalent return (DeMiguel et al. 2009), annualised: 12(mean - gamma/2 var)."""
    ex = r - rfp
    return 12 * (ex.mean() - gamma / 2 * ex.var(ddof=1))


def net_of_costs(name, bps):
    """Monthly return net of proportional costs c * turnover (turnover at t charged in t+1)."""
    r = oos[name]
    if name == "HSI":
        return r
    to = metrics.turnover(res[name])
    cost = pd.Series(0.0, index=r.index)
    nxt = {d: period[i] for i, d in enumerate(res[name].weights.index)}
    for d, v in to.items():
        cost[nxt[d]] = bps / 1e4 * v
    return r - cost


def boot_p(r1, r2, block=6, reps=5000):
    """Circular block bootstrap p-value for H0: Sharpe(r1) = Sharpe(r2) (two-sided, centred)."""
    x = pd.concat([r1, r2], axis=1).sub(rfp, axis=0).dropna().to_numpy()
    n = len(x)

    def sr(a):
        return a.mean(0) / a.std(0, ddof=1)

    d0 = np.subtract(*sr(x))
    starts = RNG_BOOT.integers(0, n, size=(reps, int(np.ceil(n / block))))
    idx = (starts[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :n] % n
    d = np.array([np.subtract(*sr(x[i])) for i in idx])
    return float(np.mean(np.abs(d - d.mean()) >= abs(d0)))


def gross(name):
    return res[name].weights.abs().sum(axis=1)


# ---------------------------------------------------------------- table formatting
def fmt(v, kind):
    if pd.isna(v):
        return "--"
    out = {
        "pct": f"{100 * v:.1f}",
        "f2": f"{v:.2f}",
        "f3": f"{v:.3f}",
        "f1": f"{v:.1f}",
        "p": f"{v:.3f}" if v >= 0.001 else "$<$0.001",
    }[kind]
    return out.replace("-", "$-$", 1) if out.startswith("-") else out


# ==========================================================================================
# Report Section 2.1 - Mean-variance framework
# ==========================================================================================


# --- Figure 1: Monte Carlo, estimation error in the efficient frontier
full = returns.dropna(axis=1)  # stocks with the full 2009-2026 history
pick = RNG_MC.choice(full.columns, size=20, replace=False)
X = full[pick].to_numpy()
mu_true, S_true = X.mean(0), np.cov(X, rowvar=False)
rf_m = float(rfp.mean())


def frontier(mu, S):
    Si = np.linalg.inv(S)
    e = np.ones(len(mu))
    A, B, C = e @ Si @ mu, mu @ Si @ mu, e @ Si @ e
    return A, B, C, Si


def tangency(mu, S):
    Si = np.linalg.inv(S)
    x = Si @ (mu - rf_m)
    return x / x.sum()


A, B, C, _ = frontier(mu_true, S_true)
z = np.linspace(0, 0.035, 200)
sig = np.sqrt((C * z**2 - 2 * A * z + B) / (B * C - A**2))
fig, ax = plt.subplots(figsize=(W1, 2.9))
ax.plot(100 * np.sqrt(12) * sig, 100 * 12 * z, color="#111", lw=2, label="True frontier")
est_pts, true_pts = [], []
for i in range(300):
    Y = RNG_MC.multivariate_normal(mu_true, S_true, size=WINDOW)
    m, S = Y.mean(0), np.cov(Y, rowvar=False)
    a, b, c, _ = frontier(m, S)
    if i < 8:
        s2 = np.sqrt(np.maximum((c * z**2 - 2 * a * z + b) / (b * c - a**2), 0))
        ax.plot(
            100 * np.sqrt(12) * s2,
            100 * 12 * z,
            color=COL["Tangent-LO"],
            lw=0.7,
            alpha=0.6,
            label="Estimated frontiers (60 months)" if i == 0 else None,
        )
    w = tangency(m, S)
    true_pts.append((np.sqrt(w @ S_true @ w), w @ mu_true, np.abs(w).sum()))
    est_pts.append((np.sqrt(w @ S @ w), w @ m))
true_pts, est_pts = np.array(true_pts), np.array(est_pts)
shown = true_pts[:, 2] < 10  # plot only portfolios with gross exposure < 10 (axis range)
w_t = tangency(mu_true, S_true)
ax.scatter(
    100 * np.sqrt(12) * true_pts[shown, 0],
    100 * 12 * true_pts[shown, 1],
    s=5,
    color="#c0392b",
    alpha=0.45,
    label="Estimated tangency portfolios, true risk/return",
)
w_e = np.ones(20) / 20
ax.scatter(
    100 * np.sqrt(12) * np.sqrt(w_e @ S_true @ w_e),
    100 * 12 * w_e @ mu_true,
    s=60,
    marker="*",
    color=COL["1/N"],
    zorder=4,
    label="1/N portfolio",
)
ax.scatter(
    100 * np.sqrt(12) * np.sqrt(w_t @ S_true @ w_t),
    100 * 12 * w_t @ mu_true,
    s=40,
    marker="D",
    color="#111",
    zorder=4,
    label="True tangency portfolio",
)
ax.set_xlim(0, 60)
ax.set_ylim(-10, 42)
ax.set_xlabel("Annualised volatility (%)")
ax.set_ylabel("Annualised mean (%)")
ax.legend(loc="lower right", frameon=False, fontsize=7)
save(fig, "mc_frontier")
sr_true_tan = (w_t @ mu_true - rf_m) / np.sqrt(w_t @ S_true @ w_t) * np.sqrt(12)
sr_ew = (w_e @ mu_true - rf_m) / np.sqrt(w_e @ S_true @ w_e) * np.sqrt(12)
sr_est_true = (true_pts[:, 1] - rf_m) / true_pts[:, 0] * np.sqrt(12)
sr_est_insample = (est_pts[:, 1] - rf_m) / est_pts[:, 0] * np.sqrt(12)
pd.Series(
    {
        "sr_true_tangency": sr_true_tan,
        "sr_1n": sr_ew,
        "sr_est_true_median": np.median(sr_est_true),
        "share_est_beats_1n": np.mean(sr_est_true > sr_ew),
        "sr_est_insample_median": np.median(sr_est_insample),
        "n_sims": len(true_pts),
        "share_shown": shown.mean(),
        "share_gross_gt3": np.mean(true_pts[:, 2] > 3),
    }
).to_csv(REPORTS / "report_mc.csv")


# --- Monte Carlo statistics quoted in the text: repeated over many random 20-stock sets, so the
#     conclusion does not hinge on one particular draw of stocks (the figure shows one set)
def sharpe_true(w, mu, S):
    return (w @ mu - rf_m) / np.sqrt(w @ S @ w) * np.sqrt(12)


mc_rows = []
for _ in range(100):
    cols = RNG_MC.choice(full.columns, size=20, replace=False)
    Xs = full[cols].to_numpy()
    mu_s, S_s = Xs.mean(0), np.cov(Xs, rowvar=False)
    ew = np.ones(20) / 20
    tan_sr, gmv_sr, ins_sr = [], [], []
    for _ in range(200):
        Y = RNG_MC.multivariate_normal(mu_s, S_s, size=WINDOW)
        m, S = Y.mean(0), np.cov(Y, rowvar=False)
        wt = tangency(m, S)
        wg = np.linalg.solve(S, np.ones(20))
        wg /= wg.sum()
        tan_sr.append(sharpe_true(wt, mu_s, S_s))
        gmv_sr.append(sharpe_true(wg, mu_s, S_s))
        ins_sr.append(sharpe_true(wt, m, S))
    tan_sr, gmv_sr = np.array(tan_sr), np.array(gmv_sr)
    mc_rows.append(
        {
            "sr_true_tangency": sharpe_true(tangency(mu_s, S_s), mu_s, S_s),
            "sr_1n": sharpe_true(ew, mu_s, S_s),
            "tan_est_median": np.median(tan_sr),
            "gmv_est_median": np.median(gmv_sr),
            "insample_median": np.median(ins_sr),
            "tan_beats_1n": np.mean(tan_sr > sharpe_true(ew, mu_s, S_s)),
            "gmv_beats_1n": np.mean(gmv_sr > sharpe_true(ew, mu_s, S_s)),
        }
    )
mc_all = pd.DataFrame(mc_rows)
mc_all.median().to_csv(REPORTS / "report_mc_repeated.csv")
mc_all.describe().to_csv(REPORTS / "report_mc_repeated_describe.csv", float_format="%.3f")


# ==========================================================================================
# Report Section 2.3 - Data
# ==========================================================================================


# --- Data overview numbers quoted in the text
fullr = returns
n_t = res["1/N"].n_assets
ann_m = 12 * fullr.mean()
ann_v = np.sqrt(12) * fullr.std()
dsum = {
    "n_tickers": fullr.shape[1],
    "months": fullr.shape[0],
    "first": fullr.index[0].strftime("%Y-%m"),
    "last": fullr.index[-1].strftime("%Y-%m"),
    "late_list": int((fullr.iloc[0].isna()).sum()),
    "n_min": int(n_t.min()),
    "n_max": int(n_t.max()),
    "n_mean": n_t.mean(),
    "med_mean": ann_m.median(),
    "med_vol": ann_v.median(),
    "hsi_mean": metrics.ann_mean(oos["HSI"]),
    "hsi_vol": metrics.ann_vol(oos["HSI"]),
    "rf_mean": 12 * rfp.mean(),
    "rf_min": 12 * rfp.min(),
    "rf_max": 12 * rfp.max(),
}
pd.Series(dsum).to_csv(REPORTS / "report_data_summary.csv")


# --- Figure 2: number of eligible stocks
fig, ax = plt.subplots(figsize=(W2, 2.1))
ax.plot(n_t.index, n_t.values, color=COL["1/N"])
ax.set_ylabel("Eligible stocks $N_t$")
ax.axhline(WINDOW, color="#999", ls=":", lw=1)
ax.text(n_t.index[2], WINDOW + 1.5, "$T=60$ (window length)", fontsize=7.5, color="#666")
ax.set_ylim(55, n_t.max() + 6)
yearaxis(ax)
save(fig, "universe_size")


# --- Figure 3: realised risk-return plane
fig, ax = plt.subplots(figsize=(W2, 2.6))
for k in LO + ["HSI"]:
    x, y = 100 * metrics.ann_vol(oos[k]), 100 * metrics.ann_mean(oos[k])
    ax.scatter(x, y, color=COL[k], s=28, zorder=3, marker="s" if k == "HSI" else "o")
    off = {"GMVP-LW-LO": (4, -9), "Tangent-LW-LO": (4, -2), "Tangent-LO": (-4, 5)}.get(k, (4, 2))
    ax.annotate(
        k,
        (x, y),
        xytext=off,
        textcoords="offset points",
        fontsize=7,
        ha="right" if off[0] < 0 else "left",
    )
x0 = 100 * 12 * rfp.mean()
for k in ["1/N", "Tangent-LO"]:
    x, y = 100 * metrics.ann_vol(oos[k]), 100 * metrics.ann_mean(oos[k])
    xs = np.array([0, 24])
    ax.plot(xs, x0 + (y - x0) / x * xs, color=COL[k], lw=0.8, ls=":")
ax.set_xlim(0, 24)
ax.set_ylim(0, 20)
ax.set_xlabel("Annualised volatility (%)")
ax.set_ylabel("Annualised mean return (%)")
save(fig, "risk_return")


# ==========================================================================================
# Report Section 3.1 - Main results
# ==========================================================================================


# --- Table 2: out-of-sample performance (long-only, 1/N, HSI)
rows = []
for k in LO + ["HSI"]:
    r = oos[k]
    rr = res.get(k)
    rows.append(
        {
            "Strategy": k,
            "mean": metrics.ann_mean(r),
            "vol": metrics.ann_vol(r),
            "sr": metrics.sharpe(r, rf),
            "ceq": ceq(r),
            "mdd": metrics.max_drawdown(r),
            "to": metrics.turnover(rr).mean() if rr else np.nan,
            "effn": metrics.effective_n(rr.weights).mean() if rr else np.nan,
            "pm": metrics.sharpe_test(r, oos["1/N"], rf) if k != "1/N" else np.nan,
            "pb": boot_p(r, oos["1/N"]) if k != "1/N" else np.nan,
        }
    )
main = pd.DataFrame(rows).set_index("Strategy")
main.to_csv(REPORTS / "report_main_table.csv", float_format="%.4f")
lines = []
for k, v in main.iterrows():
    lines.append(
        " & ".join(
            [
                k.replace("/", "/"),
                fmt(v["mean"], "pct"),
                fmt(v["vol"], "pct"),
                fmt(v["sr"], "f2"),
                fmt(v["ceq"], "pct"),
                fmt(v["mdd"], "pct"),
                fmt(v["to"], "f2"),
                fmt(v["effn"], "f1"),
                fmt(v["pm"], "p"),
                fmt(v["pb"], "p"),
            ]
        )
        + r" \\"
    )
lines.insert(len(LO), r"\midrule")
write_rows(TAB / "main.tex", lines)


# --- Figure 4: cumulative wealth
fig, ax = plt.subplots(figsize=(W1, 3.0))
ends = {}
for k in LO + ["HSI"]:
    w = (1 + oos[k]).cumprod()
    w = pd.concat([pd.Series([1.0], index=[period[0] - pd.offsets.MonthEnd(1)]), w])
    ax.plot(
        w.index,
        w.values,
        color=COL[k],
        ls="--" if k == "HSI" else "-",
        lw=2.0 if k in ("1/N", "HSI") else 1.3,
        label=k,
    )
    ends[k] = w.iloc[-1]
ypos, last = {}, None  # spread end labels so they do not overlap (log space)
for k, v in sorted(ends.items(), key=lambda kv: kv[1]):
    y = np.log(v) if last is None else max(np.log(v), last + 0.075)
    ypos[k], last = y, y
for k, v in ends.items():
    ax.annotate(
        f"{v:.2f}",
        (period[-1], v),
        xytext=(period[-1] + pd.Timedelta(days=60), np.exp(ypos[k])),
        fontsize=7,
        color=COL[k],
        va="center",
    )
ax.set_yscale("log")
ax.set_yticks([0.75, 1, 1.5, 2, 3, 4, 6])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_ylabel("Wealth of HK\\$1 invested (log scale)")
ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
yearaxis(ax)
save(fig, "cumulative")


# --- Figure 5: drawdowns
fig, ax = plt.subplots(figsize=(W1, 2.2))
for k in ["1/N", "GMVP-LO", "Tangent-LO", "HSI"]:
    w = (1 + oos[k]).cumprod()
    dd = w / np.maximum.accumulate(np.r_[1.0, w.values])[1:] - 1
    ax.plot(dd.index, 100 * dd.values, color=COL[k], ls="--" if k == "HSI" else "-", label=k)
ax.set_ylabel("Drawdown (%)")
ax.legend(ncol=4, loc="lower left", frameon=False)
yearaxis(ax)
save(fig, "drawdowns")


# ==========================================================================================
# Report Section 3.2 - What the optimisers hold
# ==========================================================================================


# --- Figure 6: average holdings (+ sector weights quoted in the text)
fig, axs = plt.subplots(1, 2, figsize=(W1, 2.8))
for a, k in zip(axs, ["GMVP-LO", "Tangent-LO"]):
    avg = res[k].weights.mean().sort_values(ascending=False).head(10)
    labels = [
        f"{meta.loc[t, 'name'][:22]} ({t.split()[0]})" if t in meta.index else t for t in avg.index
    ]
    a.barh(labels[::-1], 100 * avg.values[::-1], color=COL[k])
    a.set_title(k, loc="left")
    a.set_xlabel("Average weight (%)")
    a.tick_params(axis="y", labelsize=7)
    a.grid(axis="y", visible=False)
fig.tight_layout(w_pad=1.0)
save(fig, "holdings")
sec_rows = {}
for k in ["1/N", "GMVP-LO", "Tangent-LO", "GMVP-LW-LO"]:
    w = res[k].weights.mean()
    sec_rows[k] = w.groupby(meta["sector"].reindex(w.index).fillna("Other")).sum()
sectors = pd.DataFrame(sec_rows).fillna(0)
sectors.to_csv(REPORTS / "report_sectors.csv", float_format="%.4f")


# --- Figure 7: turnover and effective number of stocks
fig, axs = plt.subplots(1, 2, figsize=(W1, 2.1))
vals = [metrics.turnover(res[k]).mean() for k in LO]
axs[0].barh(LO[::-1], vals[::-1], color=[COL[k] for k in LO[::-1]])
axs[0].set_xlabel("Mean monthly turnover $\\sum_j|w_{j,t}-w^+_{j,t-1}|$")
vals = [metrics.effective_n(res[k].weights).mean() for k in LO]
axs[1].barh(LO[::-1], vals[::-1], color=[COL[k] for k in LO[::-1]])
axs[1].set_xlabel("Mean effective number of stocks $1/\\sum_j w_j^2$")
axs[1].set_yticklabels([])
for a in axs:
    a.grid(axis="y", visible=False)
save(fig, "turnover_effn")


# ==========================================================================================
# Report Section 3.3 - Estimation error in action
# ==========================================================================================


# --- Figure 8: in-sample promised vs realised Sharpe (Tangent-LO)
ex_ante, t_list = [], []
for t in res["Tangent-LO"].weights.index:
    w = res["Tangent-LO"].weights.loc[t]
    w = w[w > 0]
    win = window_slice(returns.loc[:t], t, WINDOW)[w.index]
    mu, cov, _ = sample_moments(win)
    r_f = float(rf.loc[period[len(t_list)]])
    ex_ante.append((w.values @ mu - r_f) / np.sqrt(w.values @ cov @ w.values) * np.sqrt(12))
    t_list.append(t)
ex_ante = pd.Series(ex_ante, index=period)
roll = (oos["Tangent-LO"] - rfp).rolling(36)
realised = roll.mean() / roll.std() * np.sqrt(12)
fig, ax = plt.subplots(figsize=(W1, 2.3))
ax.plot(
    ex_ante.index,
    ex_ante.values,
    color=COL["Tangent-LO"],
    label="In-sample Sharpe promised at each rebalance (60-month window)",
)
ax.plot(
    realised.index,
    realised.values,
    color="#222",
    lw=1.2,
    label="Realised out-of-sample Sharpe (rolling 36 months)",
)
ax.axhline(metrics.sharpe(oos["Tangent-LO"], rf), color="#222", ls=":", lw=1)
ax.axhline(0, color="#999", lw=0.6)
ax.set_ylabel("Annualised Sharpe ratio")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=7.5)
yearaxis(ax)
save(fig, "exante_vs_realised")
pd.Series(
    {
        "exante_mean": ex_ante.mean(),
        "exante_min": ex_ante.min(),
        "exante_max": ex_ante.max(),
        "realised_full": metrics.sharpe(oos["Tangent-LO"], rf),
    }
).to_csv(REPORTS / "report_exante.csv")


# --- Table 3: unconstrained closed-form portfolios
urows = []
for k in UNC:
    r = oos[k]
    g = gross(k)
    urows.append(
        " & ".join(
            [
                k,
                fmt(metrics.ann_vol(r), "pct"),
                "bankrupt" if metrics.bankrupt(r) else fmt(metrics.sharpe(r, rf), "f2"),
                fmt(metrics.max_drawdown(r), "pct"),
                fmt(metrics.turnover(res[k]).mean(), "f2"),
                fmt(g.median(), "f1"),
                f"{g.max():.0f}",
                str(int(res[k].info["singular"].sum())) if "singular" in res[k].info else "0",
                str(int(res[k].info["fallback"].sum())) if "fallback" in res[k].info else "--",
            ]
        )
        + r" \\"
    )
write_rows(TAB / "unconstrained.tex", urows)


# --- Figure 9: gross exposure of unconstrained portfolios
fig, ax = plt.subplots(figsize=(W1, 2.2))
for k in UNC:
    g = gross(k)
    ax.semilogy(g.index, g.values, color=COL[k], label=k, lw=1.1, ls="--" if "LW" in k else "-")
ax.axhline(1, color="#999", lw=0.8, ls=":")
ax.set_ylabel("Gross exposure $\\sum_j |w_j|$")
ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1))
yearaxis(ax)
save(fig, "leverage")


# --- Figure 10: Ledoit-Wolf diagnostics
info = res["GMVP-LW"].info
fig, axs = plt.subplots(1, 2, figsize=(W1, 2.1))
axs[0].plot(info.index, info["lw_delta"].astype(float), color=COL["GMVP-LW"])
axs[0].set_ylabel("Shrinkage intensity $\\delta_t$")
axs[0].set_ylim(0, 1)
axs[1].semilogy(
    info.index, info["cond_sample"].astype(float), color="#111", label="Sample $\\hat\\Sigma$"
)
axs[1].semilogy(
    info.index,
    info["cond_lw"].astype(float),
    color=COL["GMVP-LW"],
    label="Ledoit--Wolf $\\hat\\Sigma_{LW}$",
)
axs[1].set_ylabel("Condition number")
axs[1].legend(frameon=False, loc="center right")
for a in axs:
    yearaxis(a, 3)
fig.tight_layout(w_pad=1.5)
save(fig, "lw_diagnostics")


# ==========================================================================================
# Report Section 3.4 - Robustness
# ==========================================================================================


# --- Table 4: sub-periods
sub = {"2014--2019": ("2014-02", "2019-12"), "2020--2026": ("2020-01", "2026-09")}
srows = []
for k in LO + ["HSI"]:
    cells = [k]
    for a, b in sub.values():
        r = oos[k].loc[a:b]
        cells += [
            fmt(metrics.ann_mean(r), "pct"),
            fmt(metrics.ann_vol(r), "pct"),
            fmt(metrics.sharpe(r, rf), "f2"),
        ]
    srows.append(" & ".join(cells) + r" \\")
srows.insert(len(LO), r"\midrule")
write_rows(TAB / "subperiods.tex", srows)


# --- Table 5 and Figure 11: transaction costs
costs = [0, 10, 25, 50, 100]
crow = []
cost_sr = {}
for k in LO:
    cost_sr[k] = [metrics.sharpe(net_of_costs(k, c), rf) for c in costs]
    crow.append(" & ".join([k] + [f"{s:.2f}" for s in cost_sr[k]]) + r" \\")
write_rows(TAB / "costs.tex", crow)

fig, ax = plt.subplots(figsize=(W2, 2.3))
for k in LO:
    ax.plot(costs, cost_sr[k], marker="o", ms=3, color=COL[k], label=k)
ax.axhline(metrics.sharpe(oos["HSI"], rf), color=COL["HSI"], ls="--", lw=1, label="HSI")
ax.set_xlabel("One-way cost per unit turnover (bp)")
ax.set_ylabel("Sharpe ratio, net")
ax.set_ylim(0.2, 0.8)
ax.legend(frameon=False, fontsize=6.5, loc="lower left", ncol=2, bbox_to_anchor=(0, 0.06))
save(fig, "costs")


# --- Table 6 and Table 7: estimation window and universe size (from 03_sensitivity.py)
sw = pd.read_csv(REPORTS / "sensitivity_window.csv").pivot(
    index="strategy", columns="window", values="sharpe"
)
wrows = [" & ".join([k] + [f"{sw.loc[k, w]:.2f}" for w in sw.columns]) + r" \\" for k in LO]
write_rows(TAB / "sens_window.tex", wrows)
sn = pd.read_csv(REPORTS / "sensitivity_n.csv")
nrows = []
for k in LO:
    cells = [k]
    for n in (10, 20, 30):
        v = sn[(sn.N == n) & (sn.strategy == k)].iloc[0]
        cells.append(f"{v['mean']:.2f}")
        cells.append("--" if k == "1/N" else f"{100 * v['share_beats_1N']:.0f}\\%")
    nrows.append(" & ".join(cells) + r" \\")
write_rows(TAB / "sens_n.tex", nrows)


# --- Figure 12: universe size
fig, ax = plt.subplots(figsize=(W2, 2.3))
for k in LO:
    v = sn[sn.strategy == k].set_index("N")
    ax.plot(
        v.index, v["mean"], marker="o", ms=3, color=COL[k], label=k, lw=2 if k == "1/N" else 1.3
    )
ax.set_xticks([10, 20, 30])
ax.set_xlabel("Universe size $N$ (random draws, 20 per size)")
ax.set_ylabel("Mean Sharpe ratio")
ax.legend(frameon=False, fontsize=6.5, loc="upper left")
save(fig, "sens_n")


# ==========================================================================================
# Numbers quoted in the text (checked against reports/report_quotes.csv)
# ==========================================================================================
quote = {
    "hsi_sr": metrics.sharpe(oos["HSI"], rf),
    "n_months": len(period),
    "period": f"{period[0]:%Y-%m}--{period[-1]:%Y-%m}",
    "lw_delta_mean": info["lw_delta"].astype(float).mean(),
    "lw_delta_post2019": info["lw_delta"].astype(float).loc["2019-01":].mean(),
    "cond_sample_med": info["cond_sample"].astype(float).median(),
    "cond_lw_med": info["cond_lw"].astype(float).median(),
    "cond_lw_max": info["cond_lw"].astype(float).max(),
    "tan_bust": oos["Tangent"].index[oos["Tangent"] <= -1][0].strftime("%Y-%m")
    if metrics.bankrupt(oos["Tangent"])
    else "",
    "tanlw_bust": oos["Tangent-LW"].index[oos["Tangent-LW"] <= -1][0].strftime("%Y-%m")
    if metrics.bankrupt(oos["Tangent-LW"])
    else "",
    "gmvplo_names_mean": (res["GMVP-LO"].weights > 0).sum(axis=1).mean(),
    "tanlo_names_mean": (res["Tangent-LO"].weights > 0).sum(axis=1).mean(),
    "corr_1n_hsi": oos["1/N"].corr(oos["HSI"]),
}
pd.Series(quote).to_csv(REPORTS / "report_quotes.csv")
print(main.round(3).to_string())
print(pd.Series(dsum).to_string())
print(pd.Series(quote).to_string())
print(sectors.round(3).to_string())
print((REPORTS / "report_mc.csv").read_text(), (REPORTS / "report_exante.csv").read_text())
print(pd.DataFrame(cost_sr, index=costs).round(3).to_string())
