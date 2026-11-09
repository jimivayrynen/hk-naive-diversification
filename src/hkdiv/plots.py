"""Figures for the report (static PNGs, light background for LaTeX).

Colours follow the strategy *family*, so a family keeps its colour in every figure:
1/N blue, GMVP orange, tangent aqua, GMVP-LW yellow, tangent-LW magenta, HSI grey dashed.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

FAMILY_COLOR = {
    "1/N": "#2a78d6",
    "GMVP": "#eb6834",
    "Tangent": "#1baf7a",
    "GMVP-LW": "#eda100",
    "Tangent-LW": "#e87ba4",
    "HSI": INK_2,
}

LONG_ONLY_SET = ["1/N", "GMVP-LO", "Tangent-LO", "GMVP-LW-LO", "Tangent-LW-LO"]
UNCONSTRAINED_SET = ["1/N", "GMVP", "Tangent", "GMVP-LW", "Tangent-LW"]


def family(name: str) -> str:
    return name.split(" (")[0].removesuffix("-LO")


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def _lines(ax, series: dict[str, pd.Series], label_ends: bool = True) -> None:
    for name, s in series.items():
        s = s.dropna()
        dashed = name == "HSI"
        ax.plot(s.index, s.values, color=FAMILY_COLOR[family(name)], linewidth=2,
                linestyle="--" if dashed else "-", label=name)
        if label_ends and len(s):
            ax.annotate(name, (s.index[-1], s.values[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1),
              ncol=3, labelcolor=INK)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def wealth(r: pd.Series) -> tuple[pd.Series, pd.Timestamp | None]:
    """Growth of 1, cut at the first month with a return <= -100 % (bankruptcy).

    After bankruptcy cumprod is meaningless (two negative wealth factors turn positive),
    so the path ends at the last month with positive wealth. Returns (path, bankruptcy date).
    """
    bust = r.index[r <= -1]
    if len(bust):
        return (1 + r.loc[: bust[0]].iloc[:-1]).cumprod(), bust[0]
    return (1 + r).cumprod(), None


def cumulative(returns: dict[str, pd.Series], names: list[str], title: str, path: Path,
               log: bool = False) -> None:
    """Growth of 1 HKD for the given strategies (+ HSI if present).

    Bankrupt strategies end with an x marker at their last positive wealth.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    keys = names + (["HSI"] if "HSI" in returns else [])
    paths = {}
    for k in keys:
        w, bust = wealth(returns[k])
        label = f"{k} (bankrupt {bust:%Y-%m})" if bust is not None else k
        paths[label] = w
        if bust is not None and len(w):
            ax.plot(w.index[-1], w.iloc[-1], marker="x", markersize=9, mew=2,
                    color=FAMILY_COLOR[family(k)])
    _lines(ax, paths)
    if log:
        ax.set_yscale("log")
    _style(ax, title, "Wealth (start = 1)")
    ax.margins(x=0.08)
    _save(fig, path)


def small_multiples(values: dict[str, pd.Series], title: str, ylabel: str, path: Path,
                    log: bool = False, sharey: bool = True) -> None:
    """Long-only and unconstrained strategies in two panels."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=sharey)
    for ax, names, sub in zip(axes, (LONG_ONLY_SET, UNCONSTRAINED_SET),
                              ("Long-only", "Unconstrained"), strict=True):
        _lines(ax, {k: values[k] for k in names if k in values}, label_ends=False)
        if log:
            ax.set_yscale("log")
        _style(ax, f"{title}: {sub}", ylabel)
    _save(fig, path)


def single(s: pd.Series, title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(s.index, s.values, color=FAMILY_COLOR["1/N"], linewidth=2)
    _style(ax, title, ylabel)
    _save(fig, path)


def lw_diagnostics(info: pd.DataFrame, path: Path) -> None:
    """Shrinkage intensity δ and condition numbers in two panels (different scales)."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    a1.plot(info.index, info["lw_delta"].astype(float), color=FAMILY_COLOR["1/N"], linewidth=2)
    _style(a1, "Ledoit-Wolf shrinkage intensity δ", "δ")
    a2.plot(info.index, info["cond_sample"].astype(float), color=FAMILY_COLOR["GMVP"],
            linewidth=2, label="Sample Σ")
    a2.plot(info.index, info["cond_lw"].astype(float), color=FAMILY_COLOR["GMVP-LW"],
            linewidth=2, label="Ledoit-Wolf Σ")
    a2.set_yscale("log")
    a2.legend(frameon=False, fontsize=8, labelcolor=INK)
    _style(a2, "Condition number of Σ", "cond(Σ), log scale")
    _save(fig, path)
