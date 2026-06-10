"""
Forecast verification metrics.

Exposes deterministic and probabilistic metric functions with both
``snake_case`` (primary) and ``PascalCase`` (legacy alias) names.
"""

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
    brier_score,
    fair_brier_score,
    fair_crps_skill_score,
    fair_brier_skill_score,
)
