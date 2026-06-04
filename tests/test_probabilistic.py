"""Tests for pure probabilistic metric functions in performance.metrics.probabilistic."""

import numpy as np
import pytest

from performance.metrics.probabilistic import (
    quantile_loss,
    p_values_ensemble,
    p_values_probabilistic,
    crps_ensemble_integral,
    crps_probabilistic_integral,
    fair_crps_ensemble_spread,
)


# ---------------------------------------------------------------------------
# quantile_loss
# ---------------------------------------------------------------------------


class TestQuantileLoss:
    def test_perfect_forecast_zero_loss(self):
        """When forecast == observation for every quantile, loss is 0."""
        obs = np.array([[5.0], [10.0]])
        # All quantiles equal to the observation
        sims = np.tile(obs, (1, 3))
        probs = np.array([0.1, 0.5, 0.9])
        assert abs(quantile_loss(sims, probs, obs)) < 1e-10

    def test_known_scalar(self):
        """Single sample, single quantile: verify formula directly."""
        # obs=2, forecast=3, q_code = 1 - 0.9 = 0.1
        # loss = max(0.1*(3-2), 0.9*(2-3)) = max(0.1, -0.9) = 0.1
        sims = np.array([[3.0]])
        obs = np.array([[2.0]])
        probs = np.array([0.9])
        assert abs(quantile_loss(sims, probs, obs) - 0.1) < 1e-10

    def test_non_negative(self):
        rng = np.random.default_rng(7)
        sims = rng.normal(0, 1, (50, 5))
        obs = rng.normal(0, 1, (50, 1))
        probs = np.linspace(0.1, 0.9, 5)
        assert quantile_loss(sims, probs, obs) >= 0.0

    def test_overestimating_forecast(self):
        """When forecast > obs, loss driven by lower-quantile weight."""
        obs = np.zeros((10, 1))
        sims = np.ones((10, 1)) * 2.0  # constant overforecast
        probs = np.array([0.5])
        # q_code = 1 - 0.5 = 0.5; loss = max(0.5*(2-0), 0.5*(0-2)) = 1.0
        assert abs(quantile_loss(sims, probs, obs) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# p_values_ensemble
# ---------------------------------------------------------------------------


class TestPValuesEnsemble:
    def test_range_01(self):
        rng = np.random.default_rng(1)
        sims = rng.normal(0, 1, (100, 10))
        probs = np.cumsum(np.ones(10) / 10)
        targets = rng.normal(0, 1, (100, 1))
        pv = p_values_ensemble(sims, probs, targets)
        assert pv.min() >= 0.0
        assert pv.max() <= 1.0

    def test_target_below_all_members(self):
        """If target < every member, p-value should be 0."""
        sims = np.array([[1.0, 2.0, 3.0]])
        probs = np.array([1 / 3, 2 / 3, 1.0])
        target = np.array([[-10.0]])
        pv = p_values_ensemble(sims, probs, target)
        assert pv[0] == 0.0

    def test_target_above_all_members(self):
        """If target > every member, p-value should be max probability."""
        sims = np.array([[1.0, 2.0, 3.0]])
        probs = np.array([1 / 3, 2 / 3, 1.0])
        target = np.array([[100.0]])
        pv = p_values_ensemble(sims, probs, target)
        assert pv[0] == probs[-1]

    def test_target_at_median(self):
        """Target equal to median member → p-value = probability at that position."""
        sims = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        probs = np.cumsum(np.ones(5) / 5)
        target = np.array([[3.0]])  # 3rd member (0-indexed: position 2)
        pv = p_values_ensemble(sims, probs, target)
        # Strictly less-than comparison: members < 3 are [1,2] → max prob = 0.4
        assert abs(pv[0] - 0.4) < 1e-10


# ---------------------------------------------------------------------------
# p_values_probabilistic
# ---------------------------------------------------------------------------


class TestPValuesProbabilistic:
    def test_range_01(self):
        rng = np.random.default_rng(2)
        quantiles = np.sort(rng.normal(0, 1, (100, 9)), axis=1)
        probs = np.linspace(0.1, 0.9, 9)
        targets = rng.normal(0, 1, (100, 1))
        pv = p_values_probabilistic(quantiles, probs, targets)
        assert pv.min() >= 0.0
        assert pv.max() <= 1.0

    def test_interpolation_at_quantile(self):
        """p-value at Q50 value should be 0.5."""
        # Simple linear distribution: quantiles are 0.1 to 0.9 at their own levels
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        quantile_values = probs.copy()  # Q_p = p
        sims = quantile_values[np.newaxis, :]
        target = np.array([[0.5]])
        pv = p_values_probabilistic(sims, probs, target)
        assert abs(pv[0] - 0.5) < 1e-6

    def test_target_above_all_extrapolates_to_one(self):
        probs = np.array([0.1, 0.5, 0.9])
        sims = np.array([[1.0, 2.0, 3.0]])
        target = np.array([[100.0]])
        pv = p_values_probabilistic(sims, probs, target)
        assert pv[0] == 1.0


# ---------------------------------------------------------------------------
# CRPS — ensemble
# ---------------------------------------------------------------------------


class TestCRPSEnsemble:
    def _crps_simple_case(self):
        """One sample: obs=0, two-member ensemble [-1, 1], probs [0.5, 1.0]."""
        sims = np.array([[-1.0, 1.0]])
        probs = np.array([0.5, 1.0])
        targets = np.array([[0.0]])
        p_vals = p_values_ensemble(sims, probs, targets).reshape(-1, 1)
        crps = crps_ensemble_integral(sims, probs, targets, p_vals)
        # Analytic value: 0.5 (see docstring)
        return crps[0]

    def test_known_value(self):
        crps = self._crps_simple_case()
        assert abs(crps - 0.5) < 1e-6

    def test_non_negative(self):
        rng = np.random.default_rng(3)
        sims = rng.normal(0, 1, (50, 10))
        probs = np.cumsum(np.ones(10) / 10)
        targets = rng.normal(0, 1, (50, 1))
        p_vals = p_values_ensemble(sims, probs, targets).reshape(-1, 1)
        crps = crps_ensemble_integral(sims, probs, targets, p_vals)
        assert np.all(crps >= 0.0)

    def test_perfect_ensemble_near_zero(self):
        """If all members equal the observation, CRPS ≈ 0."""
        obs = np.array([[3.0]] * 20)
        sims = np.tile(obs, (1, 5))
        probs = np.cumsum(np.ones(5) / 5)
        p_vals = p_values_ensemble(sims, probs, obs).reshape(-1, 1)
        crps = crps_ensemble_integral(sims, probs, obs, p_vals)
        assert np.all(np.abs(crps) < 1e-6)


# ---------------------------------------------------------------------------
# Fair CRPS spread correction
# ---------------------------------------------------------------------------


class TestFairCRPSSpread:
    def test_non_negative(self):
        rng = np.random.default_rng(4)
        sims = rng.normal(0, 1, (30, 10))
        probs = np.cumsum(np.ones(10) / 10)
        spread = fair_crps_ensemble_spread(sims, probs)
        assert np.all(spread >= 0.0)

    def test_identical_members_zero_spread(self):
        """If all members are identical, spread correction is 0."""
        sims = np.tile(np.ones((10, 1)) * 5, (1, 5))
        probs = np.cumsum(np.ones(5) / 5)
        spread = fair_crps_ensemble_spread(sims, probs)
        assert np.all(np.abs(spread) < 1e-10)
