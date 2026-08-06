"""
Callable metric accessors bound to a :class:`ForecastPerformance` instance.

Each accessor is **callable** — ``fp.deterministic(metric, name, ...)`` and
``fp.probabilistic(metric, name, ...)`` accept a metric handle *or* its name,
exactly like before — and additionally exposes **one explicit method per
metric** so editors autocomplete the available metrics and their per-metric
keyword arguments::

    fp.deterministic.rmse("model", leadtime=lt)
    fp.probabilistic.crps("model", leadtime=lt)
    fp.probabilistic.brier_score("model", leadtime=lt, threshold=100)

``.metrics`` lists the underlying :class:`Metric` objects for iteration.
"""

from . import deterministic as _det
from . import probabilistic as _prob


class _MetricAccessor:
    """Base accessor holding a back-reference to the owning instance."""

    def __init__(self, fp):
        self._fp = fp


class DeterministicAccessor(_MetricAccessor):
    """Apply deterministic metrics. Callable, with per-metric methods."""

    #: The deterministic Metric objects, in display order.
    metrics = _det.DETERMINISTIC

    def __call__(self, metric, name, leadtime=None):
        return self._fp._apply_deterministic(metric, name, leadtime)

    def pearson(self, name, leadtime=None):
        return self(_det.pearson, name, leadtime=leadtime)

    def spearman(self, name, leadtime=None):
        return self(_det.spearman, name, leadtime=leadtime)

    def nse(self, name, leadtime=None):
        return self(_det.nse, name, leadtime=leadtime)

    def kge(self, name, leadtime=None):
        return self(_det.kge, name, leadtime=leadtime)

    def kge_prime(self, name, leadtime=None):
        return self(_det.kge_prime, name, leadtime=leadtime)

    def mae(self, name, leadtime=None):
        return self(_det.mae, name, leadtime=leadtime)

    def mse(self, name, leadtime=None):
        return self(_det.mse, name, leadtime=leadtime)

    def rmse(self, name, leadtime=None):
        return self(_det.rmse, name, leadtime=leadtime)

    def bias(self, name, leadtime=None):
        return self(_det.bias, name, leadtime=leadtime)

    def relative_bias(self, name, leadtime=None):
        return self(_det.relative_bias, name, leadtime=leadtime)

    def count(self, name, leadtime=None):
        return self(_det.count, name, leadtime=leadtime)


class ProbabilisticAccessor(_MetricAccessor):
    """Apply probabilistic metrics. Callable, with per-metric methods."""

    #: The probabilistic Metric objects, in display order.
    metrics = _prob.PROBABILISTIC

    def __call__(self, metric, name, leadtime=None, months=None, metric_kwargs=None):
        return self._fp._apply_probabilistic(
            metric, name, leadtime, months=months, metric_kwargs=metric_kwargs
        )

    def quantile_loss(self, name, leadtime=None, months=None):
        return self("quantile_loss", name, leadtime=leadtime, months=months)

    def crps(self, name, leadtime=None, months=None):
        return self("crps", name, leadtime=leadtime, months=months)

    def fair_crps(self, name, leadtime=None, months=None):
        return self("fair_crps", name, leadtime=leadtime, months=months)

    def reliability(self, name, leadtime=None, months=None):
        return self("reliability", name, leadtime=leadtime, months=months)

    def resolution(self, name, leadtime=None, months=None):
        return self("resolution", name, leadtime=leadtime, months=months)

    def resolution_relative(self, name, leadtime=None, months=None):
        return self("resolution_relative", name, leadtime=leadtime, months=months)

    def brier_score(
        self, name, leadtime=None, threshold=None, months=None, return_p_values=False
    ):
        return self(
            "brier_score",
            name,
            leadtime=leadtime,
            months=months,
            metric_kwargs={"threshold": threshold, "return_p_values": return_p_values},
        )

    def fair_brier_score(self, name, leadtime=None, threshold=None, months=None):
        return self(
            "fair_brier_score",
            name,
            leadtime=leadtime,
            months=months,
            metric_kwargs={"threshold": threshold},
        )

    def fair_crps_skill_score(
        self, name, leadtime=None, reference=None, reference_leadtime=None, months=None
    ):
        return self(
            "fair_crps_skill_score",
            name,
            leadtime=leadtime,
            months=months,
            metric_kwargs={
                "reference": reference,
                "reference_leadtime": reference_leadtime,
            },
        )

    def fair_brier_skill_score(
        self,
        name,
        leadtime=None,
        threshold=None,
        reference=None,
        reference_leadtime=None,
        months=None,
    ):
        return self(
            "fair_brier_skill_score",
            name,
            leadtime=leadtime,
            months=months,
            metric_kwargs={
                "threshold": threshold,
                "reference": reference,
                "reference_leadtime": reference_leadtime,
            },
        )
