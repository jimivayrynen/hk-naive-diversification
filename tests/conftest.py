import numpy as np
import pandas as pd
import pytest


def month_ends(n: int, start: str = "2010-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


@pytest.fixture
def random_returns() -> pd.DataFrame:
    """8 synthetic stocks, 40 months, one of them delists after month 30."""
    rng = np.random.default_rng(0)
    idx = month_ends(40)
    df = pd.DataFrame(rng.normal(0.01, 0.05, (40, 8)), index=idx,
                      columns=[f"S{i}" for i in range(8)])
    df.iloc[30:, 7] = np.nan
    return df


@pytest.fixture
def rf_const(random_returns) -> pd.Series:
    return pd.Series(0.001, index=random_returns.index)
