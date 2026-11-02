# hk-naive-diversification

IEDA3330 course project: mean-variance analysis vs. naive (1/N) diversification on Hong Kong equities.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                                # create .venv and install dependencies
uv run nbstripout --install --attributes .gitattributes  # strip notebook outputs on commit (once per clone)
uv run pytest                                          # run tests
uv run jupyter lab                                     # open notebooks
```

## Data

The data is licensed Bloomberg data and is **not** in this repository. Get the files from the
team's shared drive and place them in `data/raw/`:

| File | Contents |
|---|---|
| `data/raw/data.csv` | Monthly prices, HK equities (Bloomberg tickers, e.g. `700 HK`), 2009-01 – 2026-09 |
| `data/raw/BENCH.csv` | `HSI_Price`, `HSI_TotalReturn`, `HIBOR_1M` (annual %), same dates |

Never commit data files, notebook outputs containing data, or screenshots of raw data tables.
`.gitignore` blocks common data formats and `nbstripout` removes notebook outputs on commit.

## Structure

```
src/hkdiv/       # project code
tests/           # pytest tests
notebooks/       # exploratory analysis
data/raw/        # raw data (not committed)
data/processed/  # cleaned data (not committed)
reports/figures/ # generated figures
```
