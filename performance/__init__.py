"""
forecast_performance
====================

Evaluate deterministic, ensemble and probabilistic forecasts.

Quick start
-----------
>>> from performance import ForecastPerformance, rmse
>>> fp = ForecastPerformance(reference_series)
>>> fp.add(forecast_df, name='my_model')
>>> fp.deterministic(rmse, 'my_model')      # handle
>>> fp.deterministic('rmse', 'my_model')     # name
>>> fp.deterministic.rmse('my_model')        # discoverable accessor
"""

from .forecast_performance import ForecastPerformance   # noqa: F401
from .results import Results                            # noqa: F401
from .decorators import storedResults                  # noqa: F401
from .metrics import (                                  # noqa: F401
    Metric,
    DETERMINISTIC_METRICS, PROBABILISTIC_METRICS,
    DETERMINISTIC, PROBABILISTIC,
    # snake_case (primary)
    pearson, spearman, nse, kge, kge_prime,
    mae, mse, rmse, bias, relative_bias, count,
    # PascalCase aliases
    Pearson, Spearman, NSE, KGE, KGEprime, MAE, MSE, RMSE,
    # probabilistic
    quantile_loss,
    p_values_ensemble, p_values_probabilistic,
    crps_ensemble_integral, crps_probabilistic_integral,
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
)

__version__ = "0.5.0"
__all__ = [
    "ForecastPerformance",
    "Results",
    "storedResults",
    "Metric",
    "DETERMINISTIC_METRICS", "PROBABILISTIC_METRICS",
    "DETERMINISTIC", "PROBABILISTIC",
    # deterministic
    "pearson", "spearman", "nse", "kge", "kge_prime",
    "mae", "mse", "rmse", "bias", "relative_bias", "count",
    "Pearson", "Spearman", "NSE", "KGE", "KGEprime", "MAE", "MSE", "RMSE",
    # probabilistic
    "quantile_loss",
    "p_values_ensemble", "p_values_probabilistic",
    "crps_ensemble_integral", "crps_probabilistic_integral",
    "fair_crps_ensemble_spread",
    "crps",
    "fair_crps",
    "reliability",
    "resolution",
    "resolution_relative",
    "brier_score",
    "fair_brier_score",
    "fair_crps_skill_score",
    "fair_brier_skill_score",
]
