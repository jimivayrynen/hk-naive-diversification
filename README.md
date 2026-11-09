# hk-naive-diversification

IEDA3330 course project: does mean-variance optimisation beat naive 1/N diversification
out-of-sample on Hong Kong equities once estimation error is taken into account?
(Following DeMiguel, Garlappi & Uppal 2009.)

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                                  # create .venv and install dependencies
uv run nbstripout --install --attributes .gitattributes  # strip notebook outputs on commit (once per clone)
uv run pytest                                            # run tests
uv run ruff check .                                      # lint
```

## Data

The data is licensed Bloomberg data and is **not** in this repository. Get the files from the
team's shared drive and place them in `data/raw/`:

| File | Contents |
|---|---|
| `data/raw/data.csv` | Monthly **total return index** (dividends reinvested) per stock, Bloomberg tickers (e.g. `700 HK`), 2009-01 – 2026-09 |
| `data/raw/BENCH.csv` | `HSI_Price`, `HSI_TotalReturn` (a true total return only from 2013-02), `HIBOR_1M` (annual %), same dates |

Never commit data files, notebook outputs containing data, or screenshots of raw data tables.
`.gitignore` blocks common data formats (except the derived result tables in `reports/`) and
`nbstripout` removes notebook outputs on commit.

## Running the backtest

```bash
uv run python scripts/01_clean_data.py    # data/raw -> data/processed (returns, HSI, RF)
uv run python scripts/02_backtest.py      # main backtest -> reports/results.csv, reports/figures/
uv run python scripts/03_sensitivity.py   # window 36/60/120 and N = 10/20/30 (~ several minutes)
uv run python scripts/04_report_figures.py  # report figures (PDF) + LaTeX tables -> report/
cd report && latexmk -pdf main.tex          # build the written report report/main.pdf
```

Setup: monthly rebalancing, rolling 60-month window, first decision 2014-01 (first return
2014-02). Strategies: 1/N, GMVP and tangency portfolio (closed form and long-only), each with
the sample covariance and with Ledoit–Wolf shrinkage. See
[docs/backtest_walkthrough.md](docs/backtest_walkthrough.md) for the methodology, choices and
results (in Finnish).

## Structure

```
src/hkdiv/
  data.py         # load + clean raw data, returns, HSI and risk-free rate
  universe.py     # investable stocks at each decision date
  strategies.py   # 1/N, GMVP, tangency (closed form / long-only, sample / Ledoit-Wolf)
  backtest.py     # rolling-window out-of-sample loop
  metrics.py      # Sharpe, drawdown, turnover, effective N, Sharpe difference test
  sensitivity.py  # window-length and universe-size experiments
  plots.py        # report figures
scripts/          # thin runners, executed in order
tests/            # pytest tests (synthetic data only, no Bloomberg data needed)
docs/             # ticker reference, backtest walkthrough
reports/          # results.csv, sensitivity_*.csv, figures/ (derived results only)
data/raw/         # raw data (not committed)
data/processed/   # cleaned data (not committed)
```
