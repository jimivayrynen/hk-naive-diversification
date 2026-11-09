"""Clean raw data to processed"""
from hkdiv import data

levels = data.read_levels(data.RAW / "data.csv")
bench = data.read_levels(data.RAW / "BENCH.csv")
if not levels.index.equals(bench.index):
    raise ValueError("data.csv and BENCH.csv dates differ")

print("levels:", levels.shape)
print("bench:", bench.shape)

levels, empty, excluded = data.clean_levels(levels)
print("dropped empty columns:", empty)
for t in excluded:
    print(f"dropped {t}: {data.EXCLUDE[t]}")

returns, suspensions = data.mask_suspensions(data.to_returns(levels))
for t, first, last, n_months in suspensions:
    print(f"suspension {t}: {first.date()} - {last.date()} ({n_months} zero months -> NaN)")
bench_out = data.bench_returns(bench)
print("returns:", returns.shape, returns.index.min().date(), "-", returns.index.max().date())
print("HSI valid from:", bench_out["HSI"].first_valid_index().date(),
      f"({bench_out['HSI'].isna().sum()} price-only months set to NaN)")

n = returns.notna().sum(axis=1)
print(f"stocks with a return per month: min {n.min()}, median {int(n.median())}, max {n.max()}")
print("\nreturns beyond +-50%:")
print(data.extreme_returns(returns).to_string())

data.PROCESSED.mkdir(parents=True, exist_ok=True)
returns.to_csv(data.PROCESSED / "returns.csv")
bench_out.to_csv(data.PROCESSED / "bench.csv")
print("\nsaved to", data.PROCESSED)
