"""
Forecast verification metrics.

Every public metric is a :class:`~performance.metrics.base.Metric` — a callable
that stringifies to its own name, so it can be passed either as a handle
(``rmse``) or as a string (``"rmse"``) interchangeably, and dropped straight
into a :class:`~performance.Results` table without ``metric.__name__``.

``snake_case`` is the primary spelling; ``PascalCase`` aliases are retained for
backward compatibility.
"""

from .base import Metric  # noqa: F401

from .deterministic import (  # noqa: F401
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
    Spearman,
    NSE,
    KGE,
    KGEprime,
    MAE,
    MSE,
    RMSE,
    DETERMINISTIC,
)

from .probabilistic import (  # noqa: F401
    quantile_loss,
    p_values_ensemble,
    p_values_probabilistic,
    crps_ensemble_integral,
    crps_probabilistic_integral,
    fair_crps_ensemble_spread,
    crps,
    fair_crps,
    reliability,
    resolution,
    resolution_relative,
    brier_score,
    fair_brier_score,
    fair_crps_skill_score,
    fair_brier_skill_score,
    PROBABILISTIC,
)


def _build_registry(metrics):
    """Map every metric name *and* alias (lowercased) to its Metric object."""
    registry = {}
    for metric in metrics:
        registry[metric.__name__.lower()] = metric
        for alias in metric.aliases:
            registry[alias.lower()] = metric
    return registry


#: name/alias (lowercase) -> Metric, for resolving deterministic metrics.
DETERMINISTIC_METRICS = _build_registry(DETERMINISTIC)

#: name/alias (lowercase) -> Metric, for resolving probabilistic metrics.
PROBABILISTIC_METRICS = _build_registry(PROBABILISTIC)
