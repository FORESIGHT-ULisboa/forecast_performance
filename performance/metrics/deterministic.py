"""
Deterministic forecast error metrics.

All functions accept 1-D array-like ``data`` (forecast) and ``reference``
(observation) and return a scalar.  They are deliberately stateless so they
can be used independently of :class:`~performance.ForecastPerformance`.

PascalCase aliases are provided for backward compatibility with code that
previously called them as static methods on ``ForecastPerformance``.
"""

import numpy as np
import pandas as pd

from .base import Metric


# ---------------------------------------------------------------------------
# Correlation / efficiency metrics
# ---------------------------------------------------------------------------


def _pearson(data, reference):
    """Pearson correlation coefficient."""
    return pd.DataFrame((data, reference)).transpose().corr().iloc[1, 0]


def _spearman(data, reference):
    """Spearman rank correlation coefficient."""
    return (
        pd.DataFrame((data, reference))
        .transpose()
        .corr(method="spearman")
        .iloc[1, 0]
    )


def _nse(data, reference):
    """Nash–Sutcliffe efficiency coefficient.

    Reference: https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient
    """
    ref_mean = reference.mean()
    denom = ((reference - ref_mean) ** 2).sum()
    if denom == 0:
        return float("nan")
    return 1.0 - ((reference - data) ** 2).sum() / denom


def _kge(data, reference):
    """Kling–Gupta efficiency.

    Reference: https://en.wikipedia.org/wiki/Kling%E2%80%93Gupta_efficiency
    """
    r = pd.DataFrame((data, reference)).transpose().corr().iloc[1, 0]
    ref_mean = reference.mean()
    data_mean = data.mean()
    beta = data_mean / ref_mean
    alpha = data.std() / reference.std()
    return 1.0 - ((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2) ** 0.5


def _kge_prime(data, reference):
    """Modified Kling–Gupta efficiency (KGE').

    Reference: https://doi.org/10.1016/j.jhydrol.2012.01.011
    """
    r = pd.DataFrame((data, reference)).transpose().corr().iloc[1, 0]
    ref_mean = reference.mean()
    data_mean = data.mean()
    beta = data_mean / ref_mean
    gamma = (reference.std() / ref_mean) / (data.std() / data_mean)
    return 1.0 - ((r - 1) ** 2 + (gamma - 1) ** 2 + (beta - 1) ** 2) ** 0.5


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------


def _mae(data, reference):
    """Mean absolute error."""
    return float(np.abs(reference - data).mean())


def _mse(data, reference):
    """Mean squared error."""
    return float(np.square(reference - data).mean())


def _rmse(data, reference):
    """Root mean squared error."""
    return float(np.sqrt(np.square(reference - data).mean()))


def _bias(data, reference):
    """Additive bias (mean forecast minus mean observation)."""
    return float(data.mean() - reference.mean())


def _relative_bias(data, reference):
    """Relative bias normalised by the mean observation."""
    ref_mean = reference.mean()
    if ref_mean == 0:
        return float("nan")
    return float((data.mean() - ref_mean) / ref_mean)


def _count(data, reference):
    """Number of paired observations."""
    return int(np.asarray(reference).shape[0])


# ---------------------------------------------------------------------------
# Public metrics — Metric objects that stringify to their name.
# Each carries its PascalCase alias so it resolves from either spelling.
# ---------------------------------------------------------------------------

pearson = Metric("pearson", _pearson, kind="deterministic", aliases=("Pearson",))
spearman = Metric("spearman", _spearman, kind="deterministic", aliases=("Spearman",))
nse = Metric("nse", _nse, kind="deterministic", aliases=("NSE",))
kge = Metric("kge", _kge, kind="deterministic", aliases=("KGE",))
kge_prime = Metric(
    "kge_prime", _kge_prime, kind="deterministic", aliases=("KGEprime", "kgeprime")
)
mae = Metric("mae", _mae, kind="deterministic", aliases=("MAE",))
mse = Metric("mse", _mse, kind="deterministic", aliases=("MSE",))
rmse = Metric("rmse", _rmse, kind="deterministic", aliases=("RMSE",))
bias = Metric("bias", _bias, kind="deterministic")
relative_bias = Metric("relative_bias", _relative_bias, kind="deterministic")
count = Metric("count", _count, kind="deterministic")


# ---------------------------------------------------------------------------
# PascalCase aliases (backward compatibility) — same Metric objects.
# ---------------------------------------------------------------------------

Pearson = pearson
Spearman = spearman
NSE = nse
KGE = kge
KGEprime = kge_prime
MAE = mae
MSE = mse
RMSE = rmse


#: All public deterministic metrics, in display order.
DETERMINISTIC = [
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
]
