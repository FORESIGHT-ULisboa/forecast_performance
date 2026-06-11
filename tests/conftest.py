"""
Shared pytest fixtures used across all test modules.

Two flavours of data are provided:

**Real daily parquet datasets** (``tests/test_datasets_daily/``) — the primary
integration fixtures, loaded once per session:

* ``fp_det_daily``  — deterministic forecast (``det.parquet``)
* ``fp_ens_daily``  — 10-member ensemble (``ens.parquet``)
* ``fp_prob_daily`` — 10-quantile probabilistic forecast (``prob.parquet``)

plus the raw frames ``obs_daily`` / ``det_daily`` / ``ens_daily`` /
``prob_daily`` for plotting tests.

**Synthetic data** — a sinusoidal seasonal cycle plus white noise, kept for
unit tests where analytic bounds must hold exactly (e.g. ``CRPS == MAE`` for a
deterministic forecast):

* ``fp_simple``       — a single deterministic forecast column
* ``fp_ensemble``     — a 10-member ensemble forecast
* ``fp_probabilistic``— a 9-quantile probabilistic forecast
* ``fp_multi_leadtime``— a two-leadtime ensemble
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from performance import ForecastPerformance

# ---------------------------------------------------------------------------
# Real daily parquet datasets
# ---------------------------------------------------------------------------
DAILY_DIR = Path(__file__).parent / "test_datasets_daily"


@pytest.fixture(scope="session")
def obs_daily() -> pd.DataFrame:
    """Daily observations (reference), indexed by ``event_datetime``."""
    return pd.read_parquet(DAILY_DIR / "obs.parquet")


@pytest.fixture(scope="session")
def det_daily() -> pd.DataFrame:
    """Daily deterministic forecast in canonical long format."""
    return pd.read_parquet(DAILY_DIR / "det.parquet")


@pytest.fixture(scope="session")
def ens_daily() -> pd.DataFrame:
    """Daily 10-member ensemble forecast in canonical long format."""
    return pd.read_parquet(DAILY_DIR / "ens.parquet")


@pytest.fixture(scope="session")
def prob_daily() -> pd.DataFrame:
    """Daily 10-quantile probabilistic forecast in canonical long format."""
    return pd.read_parquet(DAILY_DIR / "prob.parquet")


@pytest.fixture
def fp_det_daily(obs_daily, det_daily) -> ForecastPerformance:
    fp = ForecastPerformance(obs_daily)
    fp.add(det_daily, name="det")
    return fp


@pytest.fixture
def fp_ens_daily(obs_daily, ens_daily) -> ForecastPerformance:
    fp = ForecastPerformance(obs_daily)
    fp.add(ens_daily, name="ens")
    return fp


@pytest.fixture
def fp_prob_daily(obs_daily, prob_daily) -> ForecastPerformance:
    fp = ForecastPerformance(obs_daily)
    fp.add(prob_daily, name="prob")
    return fp


@pytest.fixture
def daily_leadtime(prob_daily) -> pd.Timedelta:
    """A representative leadtime present in the daily datasets."""
    return sorted(prob_daily.index.get_level_values("leadtime").unique())[0]


# ---------------------------------------------------------------------------
# Synthetic data (analytic-bound unit tests)
# ---------------------------------------------------------------------------


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
    """N_MEMBERS-member ensemble with a leadtime/ensemble_member MultiIndex."""
    members = {}
    for m in range(N_MEMBERS):
        members[m] = SEASONAL + RNG.normal(0, spread, N)
    df = pd.DataFrame(members, index=DATES)
    df.columns = pd.MultiIndex.from_product(
        [[pd.Timedelta("0D")], range(N_MEMBERS)],
        names=["leadtime", "ensemble_member"],
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
        names=["leadtime", "non_exceedance"],
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
    fp.add(_make_noisy_forecast(bias=0), name="unbiased")
    fp.add(_make_noisy_forecast(bias=5), name="biased")
    return fp


@pytest.fixture
def fp_ensemble() -> ForecastPerformance:
    fp = ForecastPerformance(_make_reference())
    fp.add(_make_ensemble_forecast(), name="ens")
    return fp


@pytest.fixture
def fp_probabilistic() -> ForecastPerformance:
    fp = ForecastPerformance(_make_reference())
    fp.add(_make_probabilistic_forecast(), name="prob")
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
            names=["leadtime", "ensemble_member"],
        )
        dfs.append(df)

    combined = pd.concat(dfs, axis=1)
    fp.add(combined, name="ens_multi")
    return fp
