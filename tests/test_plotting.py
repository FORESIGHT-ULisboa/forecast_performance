"""
Tests for visualisation: the matplotlib ``qq_plot`` method and the Plotly
helpers in :mod:`performance.plotly_forecasting`.

Rendering is headless: matplotlib uses the ``Agg`` backend and Plotly figures
are inspected as data structures (``fig.data`` / ``fig.layout``) without ever
opening a browser.
"""

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from performance import plotly_forecasting as gof


# ---------------------------------------------------------------------------
# qq_plot (matplotlib)
# ---------------------------------------------------------------------------


class TestQQPlot:
    def test_returns_dataframe_without_plotting(self, fp_prob_daily, daily_leadtime):
        df = fp_prob_daily.qq_plot("prob", leadtimes=[daily_leadtime], plot=False)
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"uniform", "p_values", "leadtime"}
        assert len(df) > 0

    def test_p_values_in_unit_interval(self, fp_prob_daily, daily_leadtime):
        df = fp_prob_daily.qq_plot("prob", leadtimes=[daily_leadtime], plot=False)
        assert df["p_values"].between(0.0, 1.0).all()

    def test_plot_populates_axes(self, fp_prob_daily, daily_leadtime):
        fig, ax = plt.subplots()
        df = fp_prob_daily.qq_plot(
            "prob", leadtimes=[daily_leadtime], plot=True, ax=ax
        )
        # one curve per leadtime + the 1:1 diagonal reference line
        assert len(ax.lines) >= 2
        assert isinstance(df, pd.DataFrame)
        plt.close(fig)

    def test_accepts_external_axis(self, fp_ens_daily, daily_leadtime):
        fig, ax = plt.subplots()
        fp_ens_daily.qq_plot("ens", leadtimes=[daily_leadtime], plot=True, ax=ax)
        assert ax.get_title() == "ens"
        plt.close(fig)


# ---------------------------------------------------------------------------
# plotly_forecasting helpers (pure functions)
# ---------------------------------------------------------------------------


class TestColorHelpers:
    def test_color_sampler_count(self):
        colors = list(gof.color_sampler(5))
        assert len(colors) == 5

    def test_color_sampler_looping_repeats(self):
        colors = list(gof.color_sampler(6, color_loops=2))
        assert len(colors) == 6
        # two loops over a 3-colour base -> first and fourth match
        assert colors[0] == colors[3]

    def test_color_sampler_zero(self):
        assert list(gof.color_sampler(0)) == []

    def test_to_rgba_from_rgb(self):
        assert gof.to_rgba("rgb(10, 20, 30)", 0.5) == "rgba(10, 20, 30, 0.5)"

    def test_to_rgba_from_hex(self):
        assert gof.to_rgba("#0a141e", 0.5) == "rgba(10, 20, 30, 0.5)"

    def test_darken_rgba(self):
        out = gof.darken_rgba("rgba(100, 100, 100, 1)", factor=0.5, alpha=0.4)
        assert out == "rgba(50, 50, 50, 0.4)"


class TestObservedAndLayout:
    def test_add_observed_trace(self, obs_daily):
        fig = go.Figure()
        gof.add_observed_trace(fig, obs_daily)
        assert len(fig.data) == 1
        assert fig.data[0].name == "obs"

    def test_apply_default_layout(self):
        fig = go.Figure()
        gof.apply_default_layout(fig, yaxis_title="Q [m3/s]")
        assert fig.layout.yaxis.title.text == "Q [m3/s]"


# ---------------------------------------------------------------------------
# plotly_forecasting plotters (deterministic / probabilistic / ensemble)
# ---------------------------------------------------------------------------


@pytest.fixture
def some_production_dates(det_daily):
    return list(det_daily.index.get_level_values("production_datetime").unique()[:3])


class TestDeterministicPlots:
    def test_plot_lt_deterministic(self, det_daily, daily_leadtime):
        fig = go.Figure()
        gof.plot_lt_deterministic(fig, det_daily, leadtimes=[daily_leadtime])
        assert len(fig.data) > 0

    def test_plot_pd_deterministic(self, det_daily, some_production_dates):
        fig = go.Figure()
        gof.plot_pd_deterministic(
            fig, det_daily, production_datetimes=some_production_dates
        )
        assert len(fig.data) > 0


class TestProbabilisticPlots:
    def test_plot_lt_probabilistic(self, prob_daily, daily_leadtime):
        fig = go.Figure()
        gof.plot_lt_probabilistic(fig, prob_daily, leadtimes=[daily_leadtime])
        assert len(fig.data) > 0

    def test_plot_pd_probabilistic(self, prob_daily):
        prod = list(prob_daily.index.get_level_values("production_datetime").unique()[:2])
        fig = go.Figure()
        gof.plot_pd_probabilistic(fig, prob_daily, production_datetimes=prod)
        assert len(fig.data) > 0


class TestEnsemblePlots:
    def test_plot_pd_ensemble(self, ens_daily, daily_leadtime):
        prod = list(ens_daily.index.get_level_values("production_datetime").unique()[:2])
        # restrict to a single leadtime to keep the unstacked frame small
        sub = ens_daily[
            ens_daily.index.get_level_values("leadtime") == daily_leadtime
        ]
        fig = go.Figure()
        gof.plot_pd_ensemble(fig, sub, production_datetimes=prod)
        assert len(fig.data) > 0
