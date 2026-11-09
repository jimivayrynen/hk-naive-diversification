import numpy as np
import pandas as pd
import pytest

from hkdiv.universe import full_history, window_slice
from tests.conftest import month_ends


@pytest.fixture
def rets():
    idx = month_ends(10)
    return pd.DataFrame(
        {
            "FULL": 0.01,
            "LATE": [np.nan] * 4 + [0.01] * 6,  # listed at month 5
            "ENDS": [0.01] * 6 + [np.nan] * 4,  # delisted after month 6
            "GAP": [0.01] * 3 + [np.nan] + [0.01] * 6,
        },
        index=idx,
    )


def test_window_slice_ends_at_decision_date(rets):
    t = rets.index[5]
    win = window_slice(rets, t, 3)
    assert list(win.index) == list(rets.index[3:6])


def test_window_slice_needs_enough_history(rets):
    with pytest.raises(ValueError, match="not enough history"):
        window_slice(rets, rets.index[1], 3)


def test_full_history_rules(rets):
    assert list(full_history(rets, rets.index[2], 3)) == ["FULL", "ENDS", "GAP"]
    assert list(full_history(rets, rets.index[5], 3)) == ["FULL", "ENDS"]
    # ENDS has no return at month 7 -> not eligible once its window reaches past delisting
    assert list(full_history(rets, rets.index[7], 3)) == ["FULL", "LATE", "GAP"]


def test_full_history_exclude(rets):
    assert list(full_history(rets, rets.index[9], 3, exclude=["GAP"])) == ["FULL", "LATE"]


def test_reused_codes_excluded_by_default():
    idx = month_ends(5)
    rets = pd.DataFrame({"13 HK": 0.01, "1880 HK": 0.01, "700 HK": 0.01}, index=idx)
    assert list(full_history(rets, idx[4], 3)) == ["700 HK"]
