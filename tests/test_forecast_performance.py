"""
Integration tests for ForecastPerformance.

Covers data ingestion, metric computation, caching behaviour (storedResults),
deterministic wrapper, reliability/resolution, and cropping utilities.
"""

import unittest.mock as mock

import numpy as np
import pandas as pd
import pytest

from performance import ForecastPerformance, rmse, mae
from performance.metrics import probabilistic as prob_metrics


# ---------------------------------------------------------------------------
# _add: simulation type detection
# ---------------------------------------------------------------------------


class TestAdd:
    def test_simple_type_detected(self, fp_simple):
        assert fp_simple.simulations["unbiased"]["simulationType"] == "simple"

    def test_ensemble_type_detected(self, fp_ensemble):
        assert fp_ensemble.simulations["ens"]["simulationType"] == "ensemble"

    def test_probabilistic_type_detected(self, fp_probabilistic):
        assert fp_probabilistic.simulations["prob"]["simulationType"] == "probabilistic"

    def test_ensemble_probabilities_sum_to_one(self, fp_ensemble):
        probs = fp_ensemble.simulations["ens"]["probabilities"]
        assert abs(probs[-1] - 1.0) < 1e-10

    def test_probabilistic_probabilities_count(self, fp_probabilistic):
        probs = fp_probabilistic.simulations["prob"]["probabilities"]
        assert len(probs) == 9

    def test_results_dict_initialised(self, fp_simple):
        assert "unbiased" in fp_simple.results
        assert isinstance(fp_simple.results["unbiased"], dict)


# ---------------------------------------------------------------------------
# deterministic()
# ---------------------------------------------------------------------------


class TestDeterministic:
    def test_rmse_finite(self, fp_simple):
        result = fp_simple.deterministic(rmse, "unbiased")
        assert np.isfinite(result)
        assert result >= 0.0

    def test_biased_forecast_larger_rmse(self, fp_simple):
        rmse_unbiased = fp_simple.deterministic(rmse, "unbiased")
        rmse_biased = fp_simple.deterministic(rmse, "biased")
        assert rmse_biased > rmse_unbiased

    def test_mae_less_than_rmse(self, fp_simple):
        mae_val = fp_simple.deterministic(mae, "unbiased")
        rmse_val = fp_simple.deterministic(rmse, "unbiased")
        # MAE <= RMSE always
        assert mae_val <= rmse_val + 1e-10

    def test_static_method_equivalence(self, fp_simple):
        """ForecastPerformance.RMSE and rmse should give same result."""
        r1 = fp_simple.deterministic(ForecastPerformance.RMSE, "unbiased")
        r2 = fp_simple.deterministic(rmse, "unbiased")
        assert abs(r1 - r2) < 1e-10


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------


class TestCRPS:
    def test_ensemble_crps_positive(self, fp_ensemble):
        crps = fp_ensemble.probabilistic(
            prob_metrics.crps,
            "ens",
            leadtime=pd.Timedelta("0D"),
        )
        assert crps > 0.0

    def test_probabilistic_crps_positive(self, fp_probabilistic):
        crps = fp_probabilistic.probabilistic(
            prob_metrics.crps,
            "prob",
            leadtime=pd.Timedelta("0D"),
        )
        assert crps > 0.0

    def test_simple_crps_equals_mae(self, fp_simple):
        """For a deterministic forecast, CRPS equals MAE."""
        crps = fp_simple.probabilistic(prob_metrics.crps, "unbiased", leadtime=None)
        mae_val = fp_simple.deterministic(mae, "unbiased")
        assert abs(crps - mae_val) < 1e-10

    def test_crps_cached(self, fp_ensemble):
        """Second call should hit cached p-values for identical arguments."""
        lt = pd.Timedelta("0D")
        first = fp_ensemble.probabilistic(prob_metrics.crps, "ens", leadtime=lt)
        with mock.patch.object(
            prob_metrics,
            "p_values_ensemble",
            wraps=prob_metrics.p_values_ensemble,
        ) as mock_pv:
            second = fp_ensemble.probabilistic(prob_metrics.crps, "ens", leadtime=lt)
            mock_pv.assert_not_called()
        assert abs(first - second) < 1e-12

    def test_crps_months_bypasses_cache(self, fp_ensemble):
        """Calling with months returns a finite filtered score."""
        lt = pd.Timedelta("0D")
        fp_ensemble.probabilistic(prob_metrics.crps, "ens", leadtime=lt)
        result = fp_ensemble.probabilistic(
            prob_metrics.crps,
            "ens",
            leadtime=lt,
            months=[1, 2, 3],
        )
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# fairCRPS
# ---------------------------------------------------------------------------


class TestFairCRPS:
    def test_fair_le_crps_for_ensemble(self, fp_ensemble):
        """Fair CRPS <= CRPS for ensemble forecasts."""
        lt = pd.Timedelta("0D")
        crps = fp_ensemble.probabilistic(prob_metrics.crps, "ens", leadtime=lt)
        fair = fp_ensemble.probabilistic(prob_metrics.fair_crps, "ens", leadtime=lt)
        assert fair <= crps + 1e-10

    def test_fair_equals_crps_for_simple(self, fp_simple):
        """Fair CRPS == CRPS for deterministic forecasts."""
        crps = fp_simple.probabilistic(prob_metrics.crps, "unbiased", leadtime=None)
        fair = fp_simple.probabilistic(
            prob_metrics.fair_crps,
            "unbiased",
            leadtime=None,
        )
        assert abs(crps - fair) < 1e-10


# ---------------------------------------------------------------------------
# BrierS
# ---------------------------------------------------------------------------


class TestBrierS:
    def test_range_01(self, fp_ensemble):
        lt = pd.Timedelta("0D")
        threshold = 52.0  # above mean of reference
        bs = fp_ensemble.probabilistic(
            prob_metrics.brier_score,
            "ens",
            leadtime=lt,
            metric_kwargs={"threshold": threshold},
        )
        assert 0.0 <= bs <= 1.0

    def test_return_p_values(self, fp_ensemble):
        lt = pd.Timedelta("0D")
        bs, pv = fp_ensemble.probabilistic(
            prob_metrics.brier_score,
            "ens",
            leadtime=lt,
            metric_kwargs={"threshold": 50.0, "return_p_values": True},
        )
        assert isinstance(pv, pd.DataFrame)
        assert np.all(pv.values >= 0) and np.all(pv.values <= 1)


# ---------------------------------------------------------------------------
# reliability / resolution
# ---------------------------------------------------------------------------


class TestReliabilityResolution:
    def test_reliability_range(self, fp_ensemble):
        alpha = fp_ensemble.probabilistic(
            prob_metrics.reliability,
            "ens",
            leadtime=pd.Timedelta("0D"),
        )
        assert -1.0 <= alpha <= 1.0

    def test_resolution_positive(self, fp_ensemble):
        res = fp_ensemble.probabilistic(
            prob_metrics.resolution,
            "ens",
            leadtime=pd.Timedelta("0D"),
        )
        assert res > 0.0

    def test_resolution_relative_positive(self, fp_ensemble):
        res = fp_ensemble.probabilistic(
            "resolution_relative",
            "ens",
            leadtime=pd.Timedelta("0D"),
        )
        assert res > 0.0

    def test_iterable_leadtime_raises(self, fp_multi_leadtime):
        lt0 = pd.Timedelta("0D")
        lt1 = pd.Timedelta("1D")
        with pytest.raises(Exception):
            fp_multi_leadtime.probabilistic(
                prob_metrics.reliability,
                "ens_multi",
                leadtime=[lt0, lt1],
            )


# ---------------------------------------------------------------------------
# get_expected_value
# ---------------------------------------------------------------------------


class TestGetExpectedValue:
    def test_simple_returns_original(self, fp_simple):
        ev = fp_simple.get_expected_value("unbiased")
        assert ev.shape[1] == 1

    def test_ensemble_returns_series(self, fp_ensemble):
        ev = fp_ensemble.get_expected_value("ens", leadtime=pd.Timedelta("0D"))
        assert isinstance(ev, pd.Series)

    def test_probabilistic_returns_dataframe(self, fp_probabilistic):
        ev = fp_probabilistic.get_expected_value("prob", leadtime=pd.Timedelta("0D"))
        assert ev.shape[1] == 1


# ---------------------------------------------------------------------------
# crop_production_dates (bug fix: k0 was undefined)
# ---------------------------------------------------------------------------


class TestCropProductionDates:
    def test_crop_reduces_rows(self, fp_simple):
        original_len = len(fp_simple.reference)
        fp_simple.crop_production_dates(start="2019-01-01", end="2019-12-31")
        assert len(fp_simple.reference) < original_len

    def test_simulation_cropped(self, fp_simple):
        fp_simple.crop_production_dates(start="2019-01-01", end="2019-12-31")
        sim_index = fp_simple.simulations["unbiased"]["data"].index
        assert sim_index.min() >= pd.Timestamp("2019-01-01")
        assert sim_index.max() <= pd.Timestamp("2019-12-31")


# ---------------------------------------------------------------------------
# storedResults: caching decorator mechanics
# ---------------------------------------------------------------------------


class TestStoredResultsDecorator:
    def test_cache_miss_then_hit(self, fp_ensemble):
        lt = pd.Timedelta("0D")
        # Clear any existing cache
        fp_ensemble.results["ens"] = {}

        call_count = [0]
        original_p_values_ensemble = None

        import performance.metrics.probabilistic as _prob
        original = _prob.p_values_ensemble

        def counting_wrapper(*args, **kwargs):
            call_count[0] += 1
            return original(*args, **kwargs)

        with mock.patch.object(_prob, "p_values_ensemble", side_effect=counting_wrapper):
            fp_ensemble._p_values("ens", leadtime=lt)       # miss → compute
            fp_ensemble._p_values("ens", leadtime=lt)       # hit → no compute

        assert call_count[0] == 1  # only computed once

    def test_threshold_bypasses_cache(self, fp_ensemble):
        lt = pd.Timedelta("0D")
        fp_ensemble.results["ens"] = {}

        import performance.metrics.probabilistic as _prob
        original = _prob.p_values_ensemble
        call_count = [0]

        def counting_wrapper(*args, **kwargs):
            call_count[0] += 1
            return original(*args, **kwargs)

        with mock.patch.object(_prob, "p_values_ensemble", side_effect=counting_wrapper):
            fp_ensemble._p_values("ens", leadtime=lt, threshold=50.0)
            fp_ensemble._p_values("ens", leadtime=lt, threshold=50.0)

        assert call_count[0] == 2  # not cached — threshold bypasses


# ---------------------------------------------------------------------------
# getSimulationNames / getSimulationLeadtimes
# ---------------------------------------------------------------------------


class TestAccessors:
    def test_names(self, fp_simple):
        names = list(fp_simple.getSimulationNames())
        assert "unbiased" in names
        assert "biased" in names

    def test_leadtimes(self, fp_ensemble):
        lts = fp_ensemble.getSimulationLeadtimes("ens")
        assert pd.Timedelta("0D") in lts
