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

import numpy as np


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
