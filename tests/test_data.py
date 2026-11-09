import numpy as np
import pandas as pd
import pytest

from hkdiv import data
from tests.conftest import month_ends


def test_to_returns_does_not_fill_gaps():
    levels = pd.DataFrame({"A": [100, 110, np.nan, 121], "B": [np.nan, 10, 11, 12.1]},
                          index=month_ends(4))
    r = data.to_returns(levels)
    assert len(r) == 3
    assert r["A"].iloc[0] == pytest.approx(0.10)
    assert r["A"].iloc[1:].isna().all()  # no return across the gap
    assert r["B"].iloc[0] != r["B"].iloc[0]  # NaN: no level before listing
    assert r["B"].iloc[1:].tolist() == pytest.approx([0.1, 0.1])


def test_clean_levels_drops_empty_and_excluded():
    levels = pd.DataFrame({"5 HK": [np.nan] * 3, "13 HK": [1, 2, 3], "700 HK": [1, 2, 3]},
                          index=month_ends(3))
    out, empty, excluded = data.clean_levels(levels)
    assert list(out.columns) == ["700 HK"]
    assert empty == ["5 HK"]
    assert excluded == ["13 HK"]


def test_bench_returns_masks_price_only_months_and_lags_rf():
    bench = pd.DataFrame(
        {"HSI_Price": [100, 110, 121, 130],
         "HSI_TotalReturn": [100, 110, 122, 132],
         "HIBOR_1M": [1.2, 2.4, 3.6, 4.8]},
        index=month_ends(4),
    )
    out = data.bench_returns(bench)
    # month 2: TR level still equals price -> NaN; month 3 onwards differs -> valid
    assert np.isnan(out["HSI"].iloc[0])
    assert out["HSI"].iloc[1] == pytest.approx(122 / 110 - 1)
    # RF in month t is HIBOR of month t-1, monthly decimal
    assert out["RF"].tolist() == pytest.approx([0.001, 0.002, 0.003])


def test_check_index_rejects_duplicates_and_unsorted():
    idx = month_ends(3)
    with pytest.raises(ValueError, match="duplicate"):
        data.check_index(pd.DataFrame({"A": [1, 2, 3]}, index=idx[[0, 1, 1]]))
    with pytest.raises(ValueError, match="increasing"):
        data.check_index(pd.DataFrame({"A": [1, 2, 3]}, index=idx[[0, 2, 1]]))


def test_extreme_returns():
    r = pd.DataFrame({"A": [0.1, 0.6], "B": [-0.7, 0.0]}, index=month_ends(2))
    ext = data.extreme_returns(r)
    assert ext.tolist() == [-0.7, 0.6]


def test_mask_suspensions_only_long_zero_runs():
    r = pd.DataFrame({"A": [0.01, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.03]}, index=month_ends(8))
    out, runs = data.mask_suspensions(r, min_run=3)
    assert out["A"].isna().tolist() == [False, True, True, True, False, False, False, False]
    assert runs == [("A", r.index[1], r.index[3], 3)]
