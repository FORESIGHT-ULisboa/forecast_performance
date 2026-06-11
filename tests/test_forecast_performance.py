"""
Integration tests for ForecastPerformance.

Covers data ingestion, metric computation, caching behaviour (storedResults),
deterministic wrapper, reliability/resolution, and cropping utilities.
"""

import unittest.mock as mock
import warnings

import numpy as np
import pandas as pd
import pytest

from performance import ForecastPerformance, rmse, mae, nse, crps, fair_crps
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
# get_expected
# ---------------------------------------------------------------------------


class TestGetExpected:
    def test_simple_returns_original(self, fp_simple):
        ev = fp_simple.get_expected("unbiased")
        assert ev.shape[1] == 1

    def test_ensemble_returns_dataframe(self, fp_ensemble):
        ev = fp_ensemble.get_expected("ens", leadtime=pd.Timedelta("0D"))
        assert isinstance(ev, pd.DataFrame)
        assert ev.shape[1] == 1

    def test_probabilistic_returns_dataframe(self, fp_probabilistic):
        ev = fp_probabilistic.get_expected("prob", leadtime=pd.Timedelta("0D"))
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
        prod = sim_index.get_level_values("production_datetime")
        assert prod.min() >= pd.Timestamp("2019-01-01")
        assert prod.max() <= pd.Timestamp("2019-12-31")


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
        names = fp_simple.names()
        assert "unbiased" in names
        assert "biased" in names

    def test_leadtimes(self, fp_ensemble):
        lts = fp_ensemble.simulations["ens"]["leadtimes"]
        assert pd.Timedelta("0D") in lts


# ---------------------------------------------------------------------------
# Homogenized calling styles: handle == name == accessor method
# ---------------------------------------------------------------------------


class TestMetricCallingStyles:
    def test_deterministic_three_ways_agree(self, fp_simple):
        a = fp_simple.deterministic(rmse, "unbiased")          # handle
        b = fp_simple.deterministic("rmse", "unbiased")        # name
        c = fp_simple.deterministic.rmse("unbiased")           # accessor
        assert a == b == c

    def test_deterministic_pascalcase_name(self, fp_simple):
        a = fp_simple.deterministic("RMSE", "unbiased")        # PascalCase alias
        b = fp_simple.deterministic(rmse, "unbiased")
        assert a == b

    def test_probabilistic_three_ways_agree(self, fp_ensemble):
        lt = pd.Timedelta("0D")
        a = fp_ensemble.probabilistic(crps, "ens", leadtime=lt)
        b = fp_ensemble.probabilistic("crps", "ens", leadtime=lt)
        c = fp_ensemble.probabilistic.crps("ens", leadtime=lt)
        assert a == b == c

    def test_probabilistic_alias_resolution(self, fp_ensemble):
        lt = pd.Timedelta("0D")
        a = fp_ensemble.probabilistic("faircrps", "ens", leadtime=lt)
        b = fp_ensemble.probabilistic(fair_crps, "ens", leadtime=lt)
        assert a == b

    def test_resolution_relative_resolves(self, fp_ensemble):
        lt = pd.Timedelta("0D")
        a = fp_ensemble.probabilistic("resolution_relative", "ens", leadtime=lt)
        b = fp_ensemble.probabilistic.resolution_relative("ens", leadtime=lt)
        assert a == b

    def test_unknown_deterministic_metric_raises(self, fp_simple):
        with pytest.raises(Exception):
            fp_simple.deterministic("not_a_metric", "unbiased")

    def test_unknown_probabilistic_metric_raises(self, fp_ensemble):
        with pytest.raises(Exception):
            fp_ensemble.probabilistic("not_a_metric", "ens", leadtime=pd.Timedelta("0D"))

    def test_accessor_metrics_listing(self, fp_simple):
        names = [str(m) for m in fp_simple.deterministic.metrics]
        assert "rmse" in names and "nse" in names


# ---------------------------------------------------------------------------
# Probabilistic metrics are exposed as handle attributes on fp (like fp.RMSE)
# ---------------------------------------------------------------------------


class TestProbabilisticHandles:
    def test_handle_is_the_metric(self, fp_ensemble):
        assert str(fp_ensemble.CRPS) == "crps"
        assert str(fp_ensemble.fair_CRPS) == "fair_crps"
        assert str(fp_ensemble.quantile_loss) == "quantile_loss"

    def test_handle_matches_name_and_accessor(self, fp_ensemble):
        lt = pd.Timedelta("0D")
        via_handle = fp_ensemble.probabilistic(fp_ensemble.CRPS, "ens", leadtime=lt)
        via_name = fp_ensemble.probabilistic("crps", "ens", leadtime=lt)
        via_accessor = fp_ensemble.probabilistic.crps("ens", leadtime=lt)
        assert via_handle == via_name == via_accessor

    def test_all_handles_present(self, fp_ensemble):
        expected = {
            "CRPS",
            "fair_CRPS",
            "quantile_loss",
            "reliability",
            "resolution",
            "resolution_relative",
            "brier_score",
            "fair_brier_score",
            "fair_CRPS_skill_score",
            "fair_brier_skill_score",
        }
        assert expected <= set(dir(fp_ensemble))

    def test_handles_usable_in_metric_list(self, fp_ensemble):
        lt = pd.Timedelta("0D")
        metrics = [fp_ensemble.CRPS, fp_ensemble.fair_CRPS, "reliability"]
        values = [
            fp_ensemble.probabilistic(m, "ens", leadtime=lt) for m in metrics
        ]
        assert all(np.isfinite(v) for v in values)


# ---------------------------------------------------------------------------
# adjust_mean / adjust_scale (ensemble and probabilistic)
# ---------------------------------------------------------------------------


def _per_leadtime_means(fp, name):
    data = fp.simulations[name]["data"]
    lt_values = data.index.get_level_values("leadtime")
    return {
        lt: np.nanmean(data.loc[lt_values == lt, "values"])
        for lt in fp.simulations[name]["leadtimes"]
    }


class TestAdjust:
    def test_adjust_mean_ensemble(self, fp_ensemble):
        ref_mean = np.nanmean(fp_ensemble.reference.values)
        fp_ensemble.adjust_mean("ens")
        for m in _per_leadtime_means(fp_ensemble, "ens").values():
            assert np.isclose(m, ref_mean)

    def test_adjust_scale_ensemble(self, fp_ensemble):
        ref_mean = np.nanmean(fp_ensemble.reference.values)
        fp_ensemble.adjust_scale("ens")
        for m in _per_leadtime_means(fp_ensemble, "ens").values():
            assert np.isclose(m, ref_mean)

    def test_adjust_mean_probabilistic(self, fp_probabilistic):
        ref_mean = np.nanmean(fp_probabilistic.reference.values)
        fp_probabilistic.adjust_mean("prob")
        for m in _per_leadtime_means(fp_probabilistic, "prob").values():
            assert np.isclose(m, ref_mean)

    def test_adjust_scale_probabilistic(self, fp_probabilistic):
        ref_mean = np.nanmean(fp_probabilistic.reference.values)
        fp_probabilistic.adjust_scale("prob")
        for m in _per_leadtime_means(fp_probabilistic, "prob").values():
            assert np.isclose(m, ref_mean)

    def test_adjust_preserves_quantile_order(self, fp_probabilistic):
        """Per-group quantile values stay sorted after a (positive) correction."""
        fp_probabilistic.adjust_scale("prob")
        data = fp_probabilistic.simulations["prob"]["data"]
        ordered = data.groupby(
            level=["production_datetime", "event_datetime", "leadtime"]
        )["values"].apply(lambda s: s.is_monotonic_increasing)
        assert ordered.all()

    def test_adjust_simple_raises(self, fp_simple):
        with pytest.raises(Exception):
            fp_simple.adjust_mean("unbiased")
        with pytest.raises(Exception):
            fp_simple.adjust_scale("unbiased")


# ---------------------------------------------------------------------------
# Warning handling (warn flag + suppressed numerical warnings)
# ---------------------------------------------------------------------------


class TestWarnings:
    def test_incomplete_boundary_warns_by_default(self, fp_prob_daily, daily_leadtime):
        with pytest.warns(UserWarning):
            fp_prob_daily.probabilistic.crps("prob", leadtime=daily_leadtime)

    def test_warn_false_silences_userwarning(
        self, obs_daily, prob_daily, daily_leadtime
    ):
        fp = ForecastPerformance(obs_daily, warn=False)
        fp.add(prob_daily, name="prob")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            fp.probabilistic.crps("prob", leadtime=daily_leadtime)  # must not raise

    def test_no_runtime_warnings_leak(self, obs_daily, prob_daily, ens_daily, daily_leadtime):
        fp = ForecastPerformance(obs_daily, warn=False)
        fp.add(prob_daily, name="prob")
        fp.add(ens_daily, name="ens")
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            fp.probabilistic.crps("prob", leadtime=daily_leadtime)
            fp.probabilistic.crps("ens", leadtime=daily_leadtime)


# ---------------------------------------------------------------------------
# Metric objects stringify to their name
# ---------------------------------------------------------------------------


class TestMetricObjects:
    def test_str_is_name(self):
        assert str(rmse) == "rmse"
        assert str(crps) == "crps"

    def test_equals_name_string(self):
        assert rmse == "rmse"
        assert fair_crps == "fair_crps"

    def test_dunder_name(self):
        assert rmse.__name__ == "rmse"

    def test_callable(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert rmse(a, b) == 0.0

    def test_usable_as_results_field(self, fp_simple):
        """A Metric drops into a results table without metric.__name__."""
        from performance import Results

        r = Results("Model", "Metric")
        for metric in (rmse, mae, nse):
            r.append(
                Model="unbiased",
                Metric=metric,                      # no .__name__
                Value=fp_simple.deterministic(metric, "unbiased"),
            )
        df = r.to_pandas(index=["Model"], columns=["Metric"])
        cols = [str(c) for c in df.columns.get_level_values("Metric")]
        assert {"rmse", "mae", "nse"} <= set(cols)


# ---------------------------------------------------------------------------
# Cache behaviour: clear_cache forces recomputation
# ---------------------------------------------------------------------------


class TestCacheBehaviour:
    def test_cache_populated_after_call(self, fp_ensemble):
        lt = pd.Timedelta("0D")
        fp_ensemble.probabilistic(crps, "ens", leadtime=lt)
        # storedResults caches PIT p-values under results[name]["_p_values"].
        assert lt in fp_ensemble.results["ens"]["_p_values"]

    def test_clear_cache_forces_recompute(self, fp_ensemble):
        lt = pd.Timedelta("0D")
        fp_ensemble.probabilistic(crps, "ens", leadtime=lt)
        fp_ensemble.clear_cache("ens")
        assert fp_ensemble.results["ens"] == {}

        with mock.patch.object(
            prob_metrics,
            "p_values_ensemble",
            wraps=prob_metrics.p_values_ensemble,
        ) as mock_pv:
            fp_ensemble.probabilistic(crps, "ens", leadtime=lt)
            # cache was cleared -> must recompute
            assert mock_pv.called

    def test_clear_cache_all(self, fp_ensemble):
        lt = pd.Timedelta("0D")
        fp_ensemble.probabilistic(crps, "ens", leadtime=lt)
        fp_ensemble.clear_cache()  # no name -> clear everything
        assert fp_ensemble.results["ens"] == {}

    def test_clear_cache_unknown_raises(self, fp_ensemble):
        with pytest.raises(KeyError):
            fp_ensemble.clear_cache("missing")


# ---------------------------------------------------------------------------
# Simulation management: remove
# ---------------------------------------------------------------------------


class TestRemove:
    def test_remove_drops_simulation(self, fp_simple):
        fp_simple.remove("biased")
        assert "biased" not in fp_simple.names()
        assert "biased" not in fp_simple.results

    def test_remove_unknown_raises(self, fp_simple):
        with pytest.raises(KeyError):
            fp_simple.remove("missing")

    def test_other_simulations_survive(self, fp_simple):
        fp_simple.remove("biased")
        # remaining simulation still usable
        assert np.isfinite(fp_simple.deterministic(rmse, "unbiased"))


# ---------------------------------------------------------------------------
# Integration on the real daily parquet datasets
# ---------------------------------------------------------------------------


class TestDailyData:
    def test_types_detected(self, fp_det_daily, fp_ens_daily, fp_prob_daily):
        assert fp_det_daily.simulations["det"]["simulationType"] == "simple"
        assert fp_ens_daily.simulations["ens"]["simulationType"] == "ensemble"
        assert fp_prob_daily.simulations["prob"]["simulationType"] == "probabilistic"

    def test_deterministic_on_daily(self, fp_det_daily, daily_leadtime):
        val = fp_det_daily.deterministic.rmse("det", leadtime=daily_leadtime)
        assert np.isfinite(val) and val >= 0.0

    def test_crps_on_daily_ensemble(self, fp_ens_daily, daily_leadtime):
        val = fp_ens_daily.probabilistic.crps("ens", leadtime=daily_leadtime)
        assert np.isfinite(val) and val > 0.0

    def test_fair_le_crps_on_daily(self, fp_ens_daily, daily_leadtime):
        c = fp_ens_daily.probabilistic.crps("ens", leadtime=daily_leadtime)
        f = fp_ens_daily.probabilistic.fair_crps("ens", leadtime=daily_leadtime)
        assert f <= c + 1e-9

    def test_crps_skill_score_against_persistence(self, fp_ens_daily, daily_leadtime):
        persistence = fp_ens_daily.get_persistence(leadtimes=[daily_leadtime])
        fp_ens_daily.add(persistence, name="persistence")
        skill = fp_ens_daily.probabilistic.fair_crps_skill_score(
            "ens", leadtime=daily_leadtime, reference="persistence"
        )
        assert np.isfinite(skill)
