"""
Core mathematical routines for probabilistic forecast verification.

All functions are pure (no I/O, no global state) and operate on NumPy arrays.
They are called internally by :class:`~performance.ForecastPerformance` but
can also be used standalone.

Shapes
------
``simulations``  : ``(n_samples, n_members_or_quantiles)``
``probabilities``: ``(n_members_or_quantiles,)``  — CDF levels in [0, 1]
``targets``      : ``(n_samples, 1)``
``p_values``     : ``(n_samples, 1)``  — probability-integral-transform values
"""

import warnings

import numpy as np
from typing import Optional

from .base import Metric


# ---------------------------------------------------------------------------
# Quantile / pinball loss
# ---------------------------------------------------------------------------


def quantile_loss(
    simulations: np.ndarray, probabilities: np.ndarray, targets: np.ndarray
) -> float:
    """
    Mean quantile (pinball) loss averaged over samples and quantile levels.

    Formula (standard pinball loss at CDF level *p*)::

        loss(p) = max(p * (y - ŷ), (1-p) * (ŷ - y))

    Note: internally the code uses ``q = 1 - p`` as the weight variable
    (consistent with the pytorch-forecasting reference), which is
    mathematically equivalent.

    Parameters
    ----------
    simulations : (n_samples, n_quantiles)
    probabilities : (n_quantiles,), CDF quantile levels
    targets : (n_samples,) or (n_samples, 1)
    """
    targets = np.asarray(targets, dtype=float).reshape(-1, 1)
    simulations = np.asarray(simulations, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)

    q = np.tile(1.0 - probabilities, (simulations.shape[0], 1))  # (n, nq)
    t = np.tile(targets, (1, simulations.shape[1]))               # (n, nq)
    ql = np.maximum(q * (simulations - t), (1.0 - q) * (t - simulations))
    return float(ql.mean())


# ---------------------------------------------------------------------------
# Probability-integral transform (p-values)
# ---------------------------------------------------------------------------


def p_values_ensemble(
    simulations: np.ndarray, probabilities: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """
    PIT p-values for ensemble forecasts.

    Parameters
    ----------
    simulations : (n_samples, n_members) — need not be sorted
    probabilities : (n_members,) — cumulative probabilities (e.g. 0.1, 0.2, …, 1.0)
    targets : (n_samples, 1)

    Returns
    -------
    p_values : (n_samples,)
    """
    simulations = np.sort(np.asarray(simulations, dtype=float), axis=1)
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets, dtype=float).reshape(-1, 1)
    n = simulations.shape[0]

    targets_tiled = np.tile(targets, (1, simulations.shape[1]))
    probs_tiled = np.tile(probabilities[np.newaxis, :], (n, 1))

    # A target below every member yields an all-NaN slice (p-value 0); the
    # warning is expected and the NaN is replaced below, so silence it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        p_vals = np.nanmax(
            np.where(simulations < targets_tiled, probs_tiled, np.nan),
            axis=1,
            keepdims=True,
        )
    p_vals[np.isnan(p_vals)] = 0.0
    return p_vals.ravel()


def p_values_probabilistic(
    simulations: np.ndarray, probabilities: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """
    PIT p-values for probabilistic (quantile-based) forecasts.

    Parameters
    ----------
    simulations : (n_samples, n_quantiles) — quantile values in ascending order
    probabilities : (n_quantiles,) — corresponding CDF levels
    targets : (n_samples, 1)

    Returns
    -------
    p_values : (n_samples,)
    """
    simulations = np.asarray(simulations, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets, dtype=float).reshape(-1, 1)
    n = simulations.shape[0]

    p_vals = np.empty(n)
    for i in range(n):
        p_vals[i] = np.interp(
            targets[i, 0], simulations[i, :], probabilities, left=0.0, right=1.0
        )
    return p_vals


# ---------------------------------------------------------------------------
# CRPS integrals
# ---------------------------------------------------------------------------


def crps_ensemble_integral(
    simulations: np.ndarray,
    probabilities: np.ndarray,
    targets: np.ndarray,
    p_values: np.ndarray,
) -> np.ndarray:
    """
    Per-sample CRPS for ensemble forecasts (step-function CDF).

    Parameters
    ----------
    simulations : (n_samples, n_members)
    probabilities : (n_members,)
    targets : (n_samples, 1)
    p_values : (n_samples, 1) — PIT values returned by :func:`p_values_ensemble`

    Returns
    -------
    crps : (n_samples,)
    """
    n_samples, n_members = simulations.shape
    probs_tiled = np.tile(probabilities[np.newaxis, :], (n_samples, 1))  # (n, m)
    p_vals_tiled = np.tile(p_values, (1, n_members))                      # (n, m)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        prob_below_pval = np.nanmax(
            np.where(probs_tiled <= p_vals_tiled, probs_tiled, np.nan), axis=1
        )
    prob_below_pval[np.isnan(prob_below_pval)] = 0.0

    probs_ext = np.sort(np.c_[probs_tiled, prob_below_pval], axis=1)  # (n, m+1)
    tmp = np.c_[simulations, targets]                                   # (n, m+1)
    sims_ext = np.sort(tmp, axis=1)                                     # (n, m+1)

    # Position of the target in the sorted array
    target_col = n_members
    idxs = np.where(np.argsort(tmp, axis=1) == target_col)[1]
    idxs = np.tile(idxs[:, np.newaxis], (1, n_members + 1))
    col_range = np.tile(np.arange(n_members + 1)[np.newaxis, :], (n_samples, 1))
    heaviside = (idxs <= col_range).astype(float)

    integral = np.empty((n_samples, n_members))
    for i in range(n_members):
        integral[:, i] = (
            np.square(probs_ext[:, i] - heaviside[:, i])
            * (sims_ext[:, i + 1] - sims_ext[:, i])
        )
    return np.sum(integral, axis=1)


def crps_probabilistic_integral(
    simulations: np.ndarray,
    probabilities: np.ndarray,
    targets: np.ndarray,
    p_values: np.ndarray,
) -> np.ndarray:
    """
    Per-sample CRPS for probabilistic forecasts (piecewise-linear CDF).

    The caller is responsible for extending ``simulations`` and ``probabilities``
    to include the 0 and 1 boundary conditions before calling this function.

    Parameters
    ----------
    simulations : (n_samples, n_cols) — quantile values, boundaries included
    probabilities : (n_cols,) — CDF levels, boundaries included
    targets : (n_samples, 1)
    p_values : (n_samples, 1) — PIT values

    Returns
    -------
    crps : (n_samples,)
    """
    n_samples, n_cols = simulations.shape
    probs_tiled = np.tile(probabilities[np.newaxis, :], (n_samples, 1))  # (n, nc)
    probs_ext = np.sort(np.c_[probs_tiled, p_values], axis=1)            # (n, nc+1)

    tmp = np.c_[simulations, targets]                                     # (n, nc+1)
    sims_ext = np.sort(tmp, axis=1)

    target_col = n_cols
    idxs = np.where(np.argsort(tmp, axis=1) == target_col)[1]
    idxs = np.tile(idxs[:, np.newaxis], (1, n_cols + 1))
    col_range = np.tile(np.arange(n_cols + 1)[np.newaxis, :], (n_samples, 1))
    heaviside = (idxs <= col_range).astype(float)

    integral = np.full((n_samples, n_cols), np.nan)
    # Adjacent quantiles can coincide (x1 == x0), producing benign
    # divide-by-zero / invalid values that are discarded by the nansum below;
    # suppress those numerical warnings rather than surface them to callers.
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(n_cols):
            h = heaviside[:, i]
            y0 = np.abs(probs_ext[:, i] - h)
            y1 = np.abs(probs_ext[:, i + 1] - h)
            x0 = sims_ext[:, i]
            x1 = sims_ext[:, i + 1]
            b = (y1 - y0) / (x1 - x0)
            a = y0 - b * x0
            b[b == 0] = 1e-9
            integral[:, i] = (
                np.power(a + b * x1, 3) - np.power(a + b * x0, 3)
            ) / (3.0 * b)

    return np.nansum(integral, axis=1)


# ---------------------------------------------------------------------------
# Fair CRPS spread correction (ensemble only)
# ---------------------------------------------------------------------------


def fair_crps_ensemble_spread(
    simulations: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    """
    Per-sample ensemble spread correction for the fair CRPS.

    The fair CRPS removes the positive bias introduced by finite ensemble
    size: ``fairCRPS = CRPS - spread``.

    Parameters
    ----------
    simulations : (n_samples, n_members)
    probabilities : (n_members,)

    Returns
    -------
    spread : (n_samples,)
    """
    n_samples, n_members = simulations.shape
    sorted_sims = np.sort(simulations, axis=1)
    probs_tiled = np.tile(probabilities[np.newaxis, :], (n_samples, 1))

    spread = np.empty((n_samples, n_members - 1))
    for i in range(n_members - 1):
        spread[:, i] = (
            np.square(probs_tiled[:, i] * (1.0 - probs_tiled[:, i]))
            * (sorted_sims[:, i + 1] - sorted_sims[:, i])
        )
    return np.sum(spread, axis=1) / (n_members - 1)


# ---------------------------------------------------------------------------
# Unified single-leadtime metric wrappers
# ---------------------------------------------------------------------------


def crps(
    simulations: np.ndarray,
    probabilities: np.ndarray,
    targets: np.ndarray,
    simulation_type: str,
    p_values: Optional[np.ndarray] = None,
) -> float:
    """Return mean CRPS for one leadtime slice."""
    simulations = np.asarray(simulations, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets, dtype=float).reshape(-1, 1)

    if simulation_type == "ensemble":
        if p_values is None:
            p_values = p_values_ensemble(simulations, probabilities, targets)
        p_values = np.asarray(p_values, dtype=float).reshape(-1, 1)
        values = crps_ensemble_integral(simulations, probabilities, targets, p_values)
        return float(np.mean(values))

    if simulation_type == "probabilistic":
        if probabilities[0] != 0 or probabilities[-1] != 1:
            probs_c = probabilities.copy()
            sims_c = simulations.copy()
            if probabilities[0] != 0:
                probs_c = np.r_[0.0, probs_c]
                sims_c = np.c_[sims_c[:, 0], sims_c]
            if probabilities[-1] != 1:
                probs_c = np.r_[probs_c, 1.0]
                sims_c = np.c_[sims_c, sims_c[:, -1]]
        else:
            probs_c = probabilities
            sims_c = simulations

        if p_values is None:
            p_values = p_values_probabilistic(simulations, probabilities, targets)
        p_values = np.asarray(p_values, dtype=float).reshape(-1, 1)
        values = crps_probabilistic_integral(sims_c, probs_c, targets, p_values)
        return float(np.mean(values))

    if simulation_type == "simple":
        return float(np.mean(np.abs(simulations - targets)))

    raise Exception("Unknown simulation type: %s" % simulation_type)


def fair_crps(
    simulations: np.ndarray,
    probabilities: np.ndarray,
    targets: np.ndarray,
    simulation_type: str,
    p_values: Optional[np.ndarray] = None,
) -> float:
    """Return mean fair CRPS for one leadtime slice."""
    score = crps(
        simulations,
        probabilities,
        targets,
        simulation_type=simulation_type,
        p_values=p_values,
    )
    if simulation_type != "ensemble":
        return score

    spread = fair_crps_ensemble_spread(simulations, probabilities)
    return float(score - np.mean(spread))


def reliability(p_values: np.ndarray) -> float:
    """Return reliability alpha score from PIT p-values."""
    p_values = np.asarray(p_values, dtype=float).ravel()
    sorted_pv = np.sort(p_values)
    uniform = np.linspace(0, 1, sorted_pv.size)
    alpha_prime = np.abs(sorted_pv - uniform).mean()
    return float(1.0 - 2.0 * alpha_prime)


def resolution(
    simulations: np.ndarray,
    probabilities: np.ndarray,
    simulation_type: str,
    relative: bool = False,
) -> float:
    """Return forecast resolution (sharpness) for one leadtime slice."""
    simulations = np.asarray(simulations, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)

    if simulation_type == "probabilistic":
        probs = np.diff(np.unique(np.r_[0, probabilities, 1]))
        probs = np.tile(probs, (simulations.shape[0], 1))
        intermediate = np.c_[
            simulations[:, [0]],
            simulations[:, :-1] + np.diff(simulations, axis=1) / 2,
            simulations[:, [-1]],
        ]
        means = np.sum(probs * intermediate, axis=1)
        means_ = np.tile(means, (probs.shape[1], 1)).T
        stds = np.sqrt(np.sum(np.square(intermediate - means_) * probs, axis=1))
    elif simulation_type == "ensemble":
        means = simulations.mean(axis=1)
        stds = simulations.std(axis=1, ddof=0)
    else:
        return float(np.inf)

    if relative:
        return float(np.mean(means / stds))
    return float(np.mean(1.0 / stds))


def brier_score(
    p_values: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> float:
    """Return mean Brier score for a threshold exceedance event."""
    p_values = np.asarray(p_values, dtype=float).reshape(-1, 1)
    targets = np.asarray(targets, dtype=float).reshape(-1, 1)
    heaviside = np.heaviside(threshold - targets, 1)
    return float(np.mean(np.square(p_values - heaviside)))


def fair_brier_score(
    p_values: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    n_members: int,
) -> float:
    """Return fair Brier score for ensembles."""
    score = brier_score(p_values, targets, threshold)
    if n_members <= 1:
        return score
    p_values = np.asarray(p_values, dtype=float).reshape(-1, 1)
    correction = np.mean(p_values * (1.0 - p_values) / (n_members - 1))
    return float(score - correction)


def fair_crps_skill_score(score: float, score_reference: float) -> float:
    """Return fair CRPS skill score against a reference forecast."""
    return float(1.0 - score / score_reference)


def fair_brier_skill_score(score: float, score_reference: float) -> float:
    """Return fair Brier skill score against a reference forecast."""
    return float(1.0 - score / score_reference)


def _resolution_relative(
    simulations: np.ndarray,
    probabilities: np.ndarray,
    simulation_type: str,
) -> float:
    """Return relative forecast resolution (mean / spread)."""
    return resolution(simulations, probabilities, simulation_type, relative=True)


# ---------------------------------------------------------------------------
# Public metrics — Metric objects that stringify to their name.
#
# Wrapping happens after the implementations are defined; functions that call
# each other (e.g. fair_crps -> crps) resolve the name at call time, so they
# transparently go through Metric.__call__.
# ---------------------------------------------------------------------------

quantile_loss = Metric("quantile_loss", quantile_loss, kind="probabilistic")
crps = Metric("crps", crps, kind="probabilistic")
fair_crps = Metric("fair_crps", fair_crps, kind="probabilistic", aliases=("faircrps",))
reliability = Metric("reliability", reliability, kind="probabilistic")
resolution = Metric("resolution", resolution, kind="probabilistic")
resolution_relative = Metric(
    "resolution_relative", _resolution_relative, kind="probabilistic"
)
brier_score = Metric(
    "brier_score", brier_score, kind="probabilistic", aliases=("briers",)
)
fair_brier_score = Metric(
    "fair_brier_score", fair_brier_score, kind="probabilistic", aliases=("fairbriers",)
)
fair_crps_skill_score = Metric(
    "fair_crps_skill_score",
    fair_crps_skill_score,
    kind="probabilistic",
    aliases=("faircrpss",),
)
fair_brier_skill_score = Metric(
    "fair_brier_skill_score",
    fair_brier_skill_score,
    kind="probabilistic",
    aliases=("fairbrierss",),
)


#: All public probabilistic metrics, in display order.
PROBABILISTIC = [
    quantile_loss,
    crps,
    fair_crps,
    reliability,
    resolution,
    resolution_relative,
    brier_score,
    fair_brier_score,
    fair_crps_skill_score,
    fair_brier_skill_score,
]
