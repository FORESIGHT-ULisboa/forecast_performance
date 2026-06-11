"""
Tests for ForecastPerformance.normalize_dataframe and input validation decorators.
"""

import numpy as np
import pandas as pd
import pytest

from performance import ForecastPerformance


DATES = pd.date_range("2020-01-01", periods=10, freq="D")
LEADTIMES = [pd.Timedelta("1D"), pd.Timedelta("2D")]
QUANTILES = [0.1, 0.5, 0.9]
MEMBERS = [0, 1, 2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wide_probabilistic(alias="non_exceedance"):
    """Wide-format probabilistic DataFrame with configurable level name."""
    df = pd.DataFrame(
        np.random.default_rng(0).random((10, 6)),
        index=DATES,
        columns=pd.MultiIndex.from_product([LEADTIMES, QUANTILES], names=["leadtime", alias]),
    )
    df.index.name = "production_datetime"
    return df


def _wide_ensemble(alias="ensemble_member"):
    """Wide-format ensemble DataFrame with configurable level name."""
    df = pd.DataFrame(
        np.random.default_rng(1).random((10, 6)),
        index=DATES,
        columns=pd.MultiIndex.from_product([LEADTIMES, MEMBERS], names=["leadtime", alias]),
    )
    df.index.name = "production_datetime"
    return df


# ---------------------------------------------------------------------------
# normalize_dataframe — stacking
# ---------------------------------------------------------------------------


def test_wide_to_long_shape():
    df = _wide_probabilistic()
    long = ForecastPerformance.normalize_dataframe(df)
    assert long.shape[1] == 1
    assert long.columns[0] == "values"
    assert long.shape[0] == 10 * 2 * 3


def test_single_column_name():
    df = _wide_ensemble()
    long = ForecastPerformance.normalize_dataframe(df, value_name="Q")
    assert long.columns[0] == "Q"


def test_series_input():
    s = pd.Series(np.arange(10), index=DATES, name="obs")
    s.index.name = "production_datetime"
    long = ForecastPerformance.normalize_dataframe(s)
    assert isinstance(long, pd.DataFrame)
    assert long.shape == (10, 1)


# ---------------------------------------------------------------------------
# normalize_dataframe — alias normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alias", ["probability", "Probability", "prob", "quantile", "percentile"])
def test_non_exceedance_aliases(alias):
    df = _wide_probabilistic(alias=alias)
    long = ForecastPerformance.normalize_dataframe(df)
    assert "non_exceedance" in long.index.names


@pytest.mark.parametrize("alias", ["ensemble", "Ensemble member", "ensemble_number", "member"])
def test_ensemble_member_aliases(alias):
    df = _wide_ensemble(alias=alias)
    long = ForecastPerformance.normalize_dataframe(df)
    assert "ensemble_member" in long.index.names


@pytest.mark.parametrize("alias", ["leadtime", "Leadtime", "lead_time", "lead"])
def test_leadtime_aliases(alias):
    df = pd.DataFrame(
        np.ones((10, 2)),
        index=DATES,
        columns=pd.MultiIndex.from_product([LEADTIMES], names=[alias]),
    )
    df.index.name = "production_datetime"
    long = ForecastPerformance.normalize_dataframe(df)
    assert "leadtime" in long.index.names


# ---------------------------------------------------------------------------
# normalize_dataframe — missing datetime derivation
# ---------------------------------------------------------------------------


def test_derives_event_datetime():
    df = _wide_probabilistic()
    long = ForecastPerformance.normalize_dataframe(df)
    assert "event_datetime" in long.index.names
    prod = long.index.get_level_values("production_datetime")
    lt = long.index.get_level_values("leadtime")
    event = long.index.get_level_values("event_datetime")
    assert (event == prod + lt).all()


def test_derives_production_datetime():
    """Start from (event_datetime, leadtime) — derive production_datetime."""
    df = pd.DataFrame(
        np.ones((10, 2)),
        index=DATES,
        columns=pd.MultiIndex.from_product([LEADTIMES], names=["leadtime"]),
    )
    df.index.name = "event_datetime"
    long = ForecastPerformance.normalize_dataframe(df)
    assert "production_datetime" in long.index.names
    event = long.index.get_level_values("event_datetime")
    lt = long.index.get_level_values("leadtime")
    prod = long.index.get_level_values("production_datetime")
    assert (prod == event - lt).all()


def test_derives_leadtime():
    """Start from a long-format with production_datetime and event_datetime — derive leadtime."""
    prod = DATES
    event = DATES + pd.Timedelta("3D")
    mi = pd.MultiIndex.from_arrays([prod, event], names=["production_datetime", "event_datetime"])
    df = pd.DataFrame({"values": np.ones(10)}, index=mi)
    long = ForecastPerformance.normalize_dataframe(df)
    assert "leadtime" in long.index.names
    lt = long.index.get_level_values("leadtime")
    assert (lt == pd.Timedelta("3D")).all()


# ---------------------------------------------------------------------------
# normalize_dataframe — bad input
# ---------------------------------------------------------------------------


def test_non_dataframe_raises():
    with pytest.raises(TypeError, match="Series or DataFrame"):
        ForecastPerformance.normalize_dataframe([1, 2, 3])


# ---------------------------------------------------------------------------
# Validation decorators
# ---------------------------------------------------------------------------


def test_init_rejects_non_dataframe():
    with pytest.raises(TypeError, match="Series or DataFrame"):
        ForecastPerformance(42)


def test_init_rejects_non_datetime_index():
    df = pd.DataFrame({"x": [1, 2, 3]}, index=[0, 1, 2])
    with pytest.raises(ValueError, match="datetime"):
        ForecastPerformance(df)


def test_init_rejects_probabilistic_reference():
    mi = pd.MultiIndex.from_product(
        [DATES, QUANTILES], names=["production_datetime", "non_exceedance"]
    )
    df = pd.DataFrame({"vals": np.ones(len(mi))}, index=mi)
    with pytest.raises(ValueError, match="non_exceedance"):
        ForecastPerformance(df)


def test_add_rejects_non_dataframe():
    fp = ForecastPerformance(pd.Series(np.ones(10), index=DATES))
    with pytest.raises(TypeError, match="DataFrame"):
        fp.add([1, 2, 3], name="bad")


def test_add_rejects_non_datetime_index():
    fp = ForecastPerformance(pd.Series(np.ones(10), index=DATES))
    df = pd.DataFrame({"x": [1, 2]}, index=[0, 1])
    with pytest.raises(ValueError, match="datetime"):
        fp.add(df, name="bad")


def test_add_rejects_unknown_level_name():
    fp = ForecastPerformance(pd.Series(np.ones(10), index=DATES))
    df = pd.DataFrame(
        np.ones((10, 2)),
        index=DATES,
        columns=pd.MultiIndex.from_product([[1, 2]], names=["mystery_level"]),
    )
    with pytest.raises(ValueError, match="unrecognised"):
        fp.add(df, name="bad")
