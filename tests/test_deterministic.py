"""Tests for deterministic metric functions in performance.metrics.deterministic."""

import numpy as np
import pytest

from performance.metrics.deterministic import (
    pearson,
    spearman,
    nse,
    kge,
    kge_prime,
    mae,
    mse,
    rmse,
    bias,
    relative_bias,
    count,
    # PascalCase aliases
    Pearson,
    RMSE,
    NSE,
    KGE,
    KGEprime,
)


RNG = np.random.default_rng(0)
REF = np.arange(1.0, 11.0)          # [1, 2, …, 10]
PERFECT = REF.copy()
OVERCAST = REF + 2.0                 # constant +2 bias
NOISY = REF + RNG.normal(0, 1, 10)


# ---------------------------------------------------------------------------
# Perfect forecast
# ---------------------------------------------------------------------------


class TestPerfectForecast:
    def test_pearson_perfect(self):
        assert abs(pearson(PERFECT, REF) - 1.0) < 1e-10

    def test_spearman_perfect(self):
        assert abs(spearman(PERFECT, REF) - 1.0) < 1e-10

    def test_nse_perfect(self):
        assert abs(nse(PERFECT, REF) - 1.0) < 1e-10

    def test_kge_perfect(self):
        assert abs(kge(PERFECT, REF) - 1.0) < 1e-10

    def test_kge_prime_perfect(self):
        assert abs(kge_prime(PERFECT, REF) - 1.0) < 1e-10

    def test_mae_perfect(self):
        assert mae(PERFECT, REF) == 0.0

    def test_mse_perfect(self):
        assert mse(PERFECT, REF) == 0.0

    def test_rmse_perfect(self):
        assert rmse(PERFECT, REF) == 0.0

    def test_bias_perfect(self):
        assert bias(PERFECT, REF) == 0.0

    def test_relative_bias_perfect(self):
        assert relative_bias(PERFECT, REF) == 0.0


# ---------------------------------------------------------------------------
# Known-value tests
# ---------------------------------------------------------------------------


class TestKnownValues:
    def test_mae_constant_bias(self):
        assert abs(mae(OVERCAST, REF) - 2.0) < 1e-10

    def test_mse_constant_bias(self):
        assert abs(mse(OVERCAST, REF) - 4.0) < 1e-10

    def test_rmse_constant_bias(self):
        assert abs(rmse(OVERCAST, REF) - 2.0) < 1e-10

    def test_bias_constant_overcast(self):
        assert abs(bias(OVERCAST, REF) - 2.0) < 1e-10

    def test_relative_bias_constant_overcast(self):
        # bias/mean_ref = 2 / 5.5
        expected = 2.0 / REF.mean()
        assert abs(relative_bias(OVERCAST, REF) - expected) < 1e-10

    def test_count(self):
        assert count(PERFECT, REF) == 10

    def test_nse_constant_forecast_is_zero(self):
        # Constant forecast = mean of reference → NSE = 0
        constant = np.full_like(REF, REF.mean())
        assert abs(nse(constant, REF)) < 1e-10

    def test_pearson_uncorrelated(self):
        # Forecast = -reference is perfectly anticorrelated
        r = pearson(-REF, REF)
        assert abs(r + 1.0) < 1e-10

    def test_kge_beta_only(self):
        # data = 2 * REF: r=1 (perfect linear), alpha=2 (double std), beta=2 (double mean)
        # KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
        #      = 1 - sqrt(0 + 1 + 1) = 1 - sqrt(2)
        scale = 2.0
        data = REF * scale
        expected = 1.0 - np.sqrt(2.0)
        assert abs(kge(data, REF) - expected) < 1e-10


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_nse_zero_variance_reference(self):
        ref = np.ones(5)
        data = np.ones(5)
        result = nse(data, ref)
        assert np.isnan(result)

    def test_relative_bias_zero_mean_reference(self):
        ref = np.array([-1.0, 0.0, 1.0])
        data = ref + 1.0
        # mean_ref = 0 → undefined
        result = relative_bias(data, ref)
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# PascalCase aliases point to the same function
# ---------------------------------------------------------------------------


class TestAliases:
    def test_pearson_alias(self):
        assert Pearson is pearson

    def test_rmse_alias(self):
        assert RMSE is rmse

    def test_nse_alias(self):
        assert NSE is nse

    def test_kge_alias(self):
        assert KGE is kge

    def test_kgeprime_alias(self):
        assert KGEprime is kge_prime
