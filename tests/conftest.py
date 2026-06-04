"""
Shared pytest fixtures used across all test modules.

Three ForecastPerformance instances are created from synthetic data:

* ``fp_simple``       — a single deterministic forecast column
* ``fp_ensemble``     — a 10-member ensemble forecast
* ``fp_probabilistic``— a 9-quantile probabilistic forecast

The reference signal is a simple sinusoidal seasonal cycle plus white noise,
making analytic bounds easy to reason about.
"""

import numpy as np
import pandas as pd
import pytest

from performance import ForecastPerformance


# ---------------------------------------------------------------------------
# Deterministic random seed for reproducibility
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)

N = 365 * 3          # three years of daily data
DATES = pd.date_range("2018-01-01", periods=N, freq="D")
SEASONAL = 10 * np.sin(np.arange(N) * 2 * np.pi / 365) + 50
NOISE_OBS = RNG.normal(0, 2, N)
REFERENCE_VALUES = SEASONAL + NOISE_OBS

QUANTILE_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
N_MEMBERS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reference() -> pd.Series:
    return pd.Series(REFERENCE_VALUES, index=DATES, name="Reference")


def _make_noisy_forecast(bias: float = 0.0, scale: float = 1.0) -> pd.DataFrame:
    """Single-column deterministic forecast."""
    vals = SEASONAL + RNG.normal(bias, 2, N) * scale
    return pd.DataFrame({"Forecast": vals}, index=DATES)


def _make_ensemble_forecast(spread: float = 3.0) -> pd.DataFrame:
    """N_MEMBERS-member ensemble with a Leadtime/Ensemble member MultiIndex."""
    members = {}
    for m in range(N_MEMBERS):
        members[m] = SEASONAL + RNG.normal(0, spread, N)
    df = pd.DataFrame(members, index=DATES)
    df.columns = pd.MultiIndex.from_product(
        [[pd.Timedelta("0D")], range(N_MEMBERS)],
        names=["Leadtime", "Ensemble member"],
    )
    return df


def _make_probabilistic_forecast(spread: float = 3.0) -> pd.DataFrame:
    """9-quantile probabilistic forecast built from Gaussian distribution."""
    from scipy.stats import norm

    cols = []
    for q in QUANTILE_LEVELS:
        vals = SEASONAL + norm.ppf(q) * spread
        cols.append(vals)
    data = np.column_stack(cols)
    df = pd.DataFrame(data, index=DATES, columns=QUANTILE_LEVELS)
    df.columns = pd.MultiIndex.from_product(
        [[pd.Timedelta("0D")], QUANTILE_LEVELS],
        names=["Leadtime", "Probability"],
    )
    return df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reference_series() -> pd.Series:
    return _make_reference()


@pytest.fixture
def fp_simple() -> ForecastPerformance:
    fp = ForecastPerformance(_make_reference())
    fp.add_by_production_date(_make_noisy_forecast(bias=0), name="unbiased")
    fp.add_by_production_date(_make_noisy_forecast(bias=5), name="biased")
    return fp


@pytest.fixture
def fp_ensemble() -> ForecastPerformance:
    fp = ForecastPerformance(_make_reference())
    fp.add_by_production_date(_make_ensemble_forecast(), name="ens")
    return fp


@pytest.fixture
def fp_probabilistic() -> ForecastPerformance:
    fp = ForecastPerformance(_make_reference())
    fp.add_by_production_date(_make_probabilistic_forecast(), name="prob")
    return fp


@pytest.fixture
def fp_multi_leadtime() -> ForecastPerformance:
    """Ensemble forecast with two leadtimes (0 D and 1 D)."""
    ref = _make_reference()
    fp = ForecastPerformance(ref)

    # Build a two-leadtime ensemble DataFrame
    lt0 = pd.Timedelta("0D")
    lt1 = pd.Timedelta("1D")
    dfs = []
    for lt in [lt0, lt1]:
        members = {}
        for m in range(N_MEMBERS):
            members[m] = SEASONAL + RNG.normal(0, 3, N)
        df = pd.DataFrame(members, index=DATES)
        df.columns = pd.MultiIndex.from_product(
            [[lt], range(N_MEMBERS)],
            names=["Leadtime", "Ensemble member"],
        )
        dfs.append(df)

    combined = pd.concat(dfs, axis=1)
    fp.add_by_production_date(combined, name="ens_multi")
    return fp
