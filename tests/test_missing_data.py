"""
Robustness to *incomplete* data.

Real forecast archives are rarely complete: production dates are skipped,
some leadtimes/ensemble members/quantile levels are missing, and observations
have gaps. These tests punch holes in the daily datasets along every index
dimension and assert that metrics still return finite numbers and that the
Plotly helpers still render (the alignment is an inner-join on
``event_datetime`` with ``dropna``, so partial overlap must degrade gracefully
rather than blow up).
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from performance import ForecastPerformance
from performance import plotly_forecasting as gof


# ---------------------------------------------------------------------------
# Helpers — drop slices of a canonical long-format frame
# ---------------------------------------------------------------------------


def _drop_level_values(df, level, drop):
    """Return *df* with the given *drop* values removed from index *level*."""
    keep = ~df.index.get_level_values(level).isin(drop)
    return df[keep]


def _every_other(df, level):
    """Keep only every other unique value of an index *level*."""
    vals = list(df.index.get_level_values(level).unique())
    drop = vals[1::2]
    return _drop_level_values(df, level, drop)


def _inject_nans(df, step=10):
    """Set every *step*-th value to NaN (ragged data)."""
    out = df.copy()
    out.iloc[::step, 0] = np.nan
    return out


# ---------------------------------------------------------------------------
# Metrics with incomplete data
# ---------------------------------------------------------------------------


class TestMetricsMissingProductionDates:
    def test_deterministic_partial_production(self, obs_daily, det_daily, daily_leadtime):
        det = _every_other(det_daily, "production_datetime")
        fp = ForecastPerformance(obs_daily)
        fp.add(det, name="det")
        val = fp.deterministic.rmse("det", leadtime=daily_leadtime)
        assert np.isfinite(val)

    def test_crps_partial_production(self, obs_daily, ens_daily, daily_leadtime):
        ens = _every_other(ens_daily, "production_datetime")
        fp = ForecastPerformance(obs_daily)
        fp.add(ens, name="ens")
        val = fp.probabilistic.crps("ens", leadtime=daily_leadtime)
        assert np.isfinite(val) and val > 0


class TestMetricsMissingObservations:
    def test_crps_with_gappy_observations(self, obs_daily, ens_daily, daily_leadtime):
        obs = obs_daily.copy()
        obs.iloc[::7] = np.nan          # ~14 % of observations missing
        fp = ForecastPerformance(obs)
        fp.add(ens_daily, name="ens")
        val = fp.probabilistic.crps("ens", leadtime=daily_leadtime)
        assert np.isfinite(val) and val > 0

    def test_deterministic_with_gappy_observations(
        self, obs_daily, det_daily, daily_leadtime
    ):
        obs = obs_daily.copy()
        obs.iloc[::7] = np.nan
        fp = ForecastPerformance(obs)
        fp.add(det_daily, name="det")
        assert np.isfinite(fp.deterministic.mae("det", leadtime=daily_leadtime))


class TestMetricsMissingLeadtimes:
    def test_remaining_leadtime_still_scores(self, obs_daily, ens_daily):
        leadtimes = sorted(ens_daily.index.get_level_values("leadtime").unique())
        keep_lt, drop_lt = leadtimes[0], leadtimes[1:3]
        ens = _drop_level_values(ens_daily, "leadtime", drop_lt)
        fp = ForecastPerformance(obs_daily)
        fp.add(ens, name="ens")
        assert keep_lt in fp.simulations["ens"]["leadtimes"]
        for d in drop_lt:
            assert d not in fp.simulations["ens"]["leadtimes"]
        assert np.isfinite(fp.probabilistic.crps("ens", leadtime=keep_lt))


class TestMetricsMissingEnsembleMembers:
    def test_fewer_members(self, obs_daily, ens_daily, daily_leadtime):
        """Drop members 3 and 7 entirely — probabilities are recomputed."""
        ens = _drop_level_values(ens_daily, "ensemble_member", [3, 7])
        fp = ForecastPerformance(obs_daily)
        fp.add(ens, name="ens")
        n = ens.index.get_level_values("ensemble_member").nunique()
        assert len(fp.simulations["ens"]["probabilities"]) == n
        assert np.isfinite(fp.probabilistic.crps("ens", leadtime=daily_leadtime))

    def test_ragged_member_nans(self, obs_daily, ens_daily, daily_leadtime):
        """Scattered NaNs inside the ensemble must not produce NaN scores."""
        ens = _inject_nans(ens_daily, step=11)
        fp = ForecastPerformance(obs_daily)
        fp.add(ens, name="ens")
        assert np.isfinite(fp.probabilistic.crps("ens", leadtime=daily_leadtime))


class TestMetricsMissingQuantiles:
    def test_fewer_non_exceedances(self, obs_daily, prob_daily, daily_leadtime):
        levels = sorted(prob_daily.index.get_level_values("non_exceedance").unique())
        keep = levels[1:-1]   # drop the most extreme low and high quantiles
        prob = _drop_level_values(
            prob_daily, "non_exceedance", [levels[0], levels[-1]]
        )
        fp = ForecastPerformance(obs_daily)
        fp.add(prob, name="prob")
        assert len(fp.simulations["prob"]["probabilities"]) == len(keep)
        assert np.isfinite(fp.probabilistic.crps("prob", leadtime=daily_leadtime))
        assert np.isfinite(
            fp.probabilistic.quantile_loss("prob", leadtime=daily_leadtime)
        )


# ---------------------------------------------------------------------------
# Plots with incomplete data
# ---------------------------------------------------------------------------


class TestPlotsMissingData:
    def test_lt_deterministic_partial_production(self, det_daily, daily_leadtime):
        det = _every_other(det_daily, "production_datetime")
        fig = go.Figure()
        gof.plot_lt_deterministic(fig, det, leadtimes=[daily_leadtime])
        assert len(fig.data) > 0

    def test_lt_probabilistic_missing_quantiles(self, prob_daily, daily_leadtime):
        levels = sorted(prob_daily.index.get_level_values("non_exceedance").unique())
        prob = _drop_level_values(
            prob_daily, "non_exceedance", [levels[0], levels[-1]]
        )
        fig = go.Figure()
        gof.plot_lt_probabilistic(fig, prob, leadtimes=[daily_leadtime])
        assert len(fig.data) > 0

    def test_pd_ensemble_fewer_members(self, ens_daily, daily_leadtime):
        prod = list(ens_daily.index.get_level_values("production_datetime").unique()[:2])
        sub = ens_daily[ens_daily.index.get_level_values("leadtime") == daily_leadtime]
        sub = _drop_level_values(sub, "ensemble_member", [3, 7])
        fig = go.Figure()
        gof.plot_pd_ensemble(fig, sub, production_datetimes=prod)
        assert len(fig.data) > 0

    def test_plots_tolerate_nans(self, prob_daily, daily_leadtime):
        prob = _inject_nans(prob_daily, step=9)
        fig = go.Figure()
        gof.plot_lt_probabilistic(fig, prob, leadtimes=[daily_leadtime])
        assert len(fig.data) > 0
