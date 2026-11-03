"""Clean raw data to processed"""
import numpy as np
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW=ROOT/ "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

prices = pd.read_csv(RAW / "data.csv", index_col="Date", parse_dates=True)
bench = pd.read_csv(RAW / "BENCH.csv", index_col="Date", parse_dates= True)

print("prices:", prices.shape)
print("bench:", bench.shape)

# 1. Drop tickers with no data at all (e.g. 5 HK until it is re-downloaded)
empty = prices.columns[prices.isna().all()]
prices = prices.drop(columns=empty)
print("dropped empty columns:", list(empty))

# Note: 13 HK (HUTCHMED) and 1880 HK (China Tourism Group Duty Free) are reused codes
# (formerly Hutchison Whampoa / Belle). Kept on purpose: the universe is large HK stocks,
# not historical HSI membership.

# 2. Monthly simple returns. No forward-filling: a stock gets returns only
#    between its first and last available price (late listings, delistings).
returns = prices.pct_change(fill_method=None)

# 3. Benchmark: HSI total return, and 1M HIBOR as monthly risk-free rate.
#    HIBOR observed at the end of month t-1 is the rate earned during month t.
bench_out = pd.DataFrame(
    {
        "HSI": bench["HSI_TotalReturn"].pct_change(fill_method=None),
        "RF": bench["HIBOR_1M"].shift(1) / 100 / 12,
    }
)

# HSI_TotalReturn is identical to HSI_Price for 2009-01 - 2013-01 (no dividends in the
# Bloomberg series before that), so those "total returns" are really price returns.
# Set them to NaN so the benchmark is never compared on a price-only basis against
# dividend-adjusted stock returns. Benchmark comparisons are valid from 2013-02 onwards.
price_only = np.isclose(bench["HSI_TotalReturn"], bench["HSI_Price"])
bench_out.loc[price_only, "HSI"] = np.nan
print("HSI set to NaN (price-only TR):", price_only.sum(), "months, valid from",
      bench_out["HSI"].first_valid_index().date())

# First month has no previous price -> no return
returns = returns.iloc[1:]
bench_out = bench_out.iloc[1:]

print("returns:", returns.shape, returns.index.min().date(), "-", returns.index.max().date())
print(bench_out.describe())

# 4. Save
PROCESSED.mkdir(parents=True, exist_ok=True)
returns.to_csv(PROCESSED / "returns.csv")
bench_out.to_csv(PROCESSED / "bench.csv")
print("saved to", PROCESSED)
