"""
ForecastPerformance - main class for evaluating deterministic and
probabilistic weather / hydrological forecasts.
"""

import functools
import warnings
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from .decorators import storedResults
from .metrics import deterministic as _det
from .metrics import probabilistic as _prob

# ---------------------------------------------------------------------------
# Canonical MultiIndex level names and aliases
# ---------------------------------------------------------------------------

#: Mapping from any recognised alias (lowercase, underscores) to the
#: canonical level name used throughout the class.
_CANONICAL_NAMES = {
    # production datetime
    "production_datetime": "production_datetime",
    "production": "production_datetime",
    "production_date": "production_datetime",
    # event datetime
    "event_datetime": "event_datetime",
    "event": "event_datetime",
    "event_date": "event_datetime",
    # leadtime
    "leadtime": "leadtime",
    "lead_time": "leadtime",
    "lead": "leadtime",
    # probabilistic (non-exceedance probability)
    "non_exceedance": "non_exceedance",
    "probability": "non_exceedance",
    "prob": "non_exceedance",
    "quantile": "non_exceedance",
    "percentile": "non_exceedance",
    # ensemble
    "ensemble_member": "ensemble_member",
    "ensemble": "ensemble_member",
    "ensemble_number": "ensemble_member",
    "member": "ensemble_member",
}

# Preferred level order in the stored MultiIndex.
_LEVEL_ORDER = [
    "production_datetime",
    "event_datetime",
    "leadtime",
    "non_exceedance",
    "ensemble_member",
]


def _normalise_name(raw):
    """Return the canonical form of a level name, or a cleaned key if unknown."""
    if raw is None:
        return None
    key = str(raw).strip().lower().replace(" ", "_")
    return _CANONICAL_NAMES.get(key, key)


# ---------------------------------------------------------------------------
# Input validation decorators
# ---------------------------------------------------------------------------


def _validate_reference_input(func):
    """Decorator that validates the *reference* argument passed to __init__."""

    @functools.wraps(func)
    def wrapper(self, reference, *args, **kwargs):
        if not isinstance(reference, (pd.Series, pd.DataFrame)):
            raise TypeError(
                "reference must be a pandas Series or DataFrame, "
                "got %s." % type(reference).__name__
            )
        idx = reference.index
        dtypes = (
            [idx.get_level_values(i).dtype for i in range(idx.nlevels)]
            if isinstance(idx, pd.MultiIndex)
            else [idx.dtype]
        )
        if not any(
            str(d).startswith("datetime") or str(d).startswith("timedelta")
            for d in dtypes
        ):
            raise ValueError(
                "reference index must contain a datetime level; "
                "got index dtype(s): %s." % [str(d) for d in dtypes]
            )
        _prob_ens = {"non_exceedance", "ensemble_member"}
        if isinstance(idx, pd.MultiIndex):
            bad = [n for n in idx.names if _normalise_name(n) in _prob_ens]
            if bad:
                raise ValueError(
                    "reference observations must not have probabilistic or "
                    "ensemble index levels; found: %s." % bad
                )
        if isinstance(reference, pd.DataFrame) and isinstance(
            reference.columns, pd.MultiIndex
        ):
            bad = [
                n
                for n in reference.columns.names
                if _normalise_name(n) in _prob_ens
            ]
            if bad:
                raise ValueError(
                    "reference observations must not carry probabilistic or "
                    "ensemble column levels; found: %s." % bad
                )
        return func(self, reference, *args, **kwargs)

    return wrapper


def _validate_simulation_input(func):
    """Decorator that validates the *data* argument passed to add."""

    @functools.wraps(func)
    def wrapper(self, data, *args, **kwargs):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "simulation data must be a pandas DataFrame, "
                "got %s." % type(data).__name__
            )
        idx = data.index
        dtypes = (
            [idx.get_level_values(i).dtype for i in range(idx.nlevels)]
            if isinstance(idx, pd.MultiIndex)
            else [idx.dtype]
        )
        if not any(str(d).startswith("datetime") for d in dtypes):
            raise ValueError(
                "simulation data index must contain a datetime level; "
                "got index dtype(s): %s." % [str(d) for d in dtypes]
            )
        if isinstance(data.columns, pd.MultiIndex):
            unknown = [
                raw
                for raw in data.columns.names
                if raw is not None
                and _normalise_name(raw) not in _CANONICAL_NAMES.values()
            ]
            if unknown:
                raise ValueError(
                    "simulation data contains unrecognised column level name(s): %s.  "
                    "Recognised names are: %s."
                    % (unknown, sorted(_CANONICAL_NAMES.keys()))
                )
        return func(self, data, *args, **kwargs)

    return wrapper


class ForecastPerformance:
    """
    Evaluate deterministic, ensemble and probabilistic forecasts against a
    reference (observation) time series.

    Parameters
    ----------
    reference : pd.Series or pd.DataFrame
        Observed values.  A single-column DataFrame or a Series indexed by
        the observation time (``event_datetime``).

    All simulation data passed to :meth:`add` and returned by baseline
    helpers follow the **canonical long format**: a single-column
    ``pd.DataFrame`` whose row index is a ``pd.MultiIndex`` with levels

    * ``production_datetime`` — time the forecast was issued
    * ``event_datetime``      — time to which the forecast refers
    * ``leadtime``            — ``event_datetime - production_datetime``
    * ``non_exceedance``      — for probabilistic forecasts
    * ``ensemble_member``     — for ensemble forecasts

    The :meth:`normalize_dataframe` static method converts any reasonable
    wide or partially-labelled DataFrame to this canonical format.
    """

    # ------------------------------------------------------------------
    # Deterministic metrics exposed as static methods (backward-compat.)
    # ------------------------------------------------------------------
    Pearson = staticmethod(_det.pearson)
    Spearman = staticmethod(_det.spearman)
    NSE = staticmethod(_det.nse)
    KGE = staticmethod(_det.kge)
    KGEprime = staticmethod(_det.kge_prime)
    MAE = staticmethod(_det.mae)
    MSE = staticmethod(_det.mse)
    RMSE = staticmethod(_det.rmse)
    bias = staticmethod(_det.bias)
    relative_bias = staticmethod(_det.relative_bias)
    count = staticmethod(_det.count)

    # ------------------------------------------------------------------
    # Construction / data ingestion
    # ------------------------------------------------------------------

    @_validate_reference_input
    def __init__(self, reference):
        self.reference = reference
        if not isinstance(self.reference, pd.DataFrame):
            self.reference = self.reference.to_frame()
        self.reference.columns = ["Reference"]
        if not isinstance(self.reference.index, pd.MultiIndex):
            self.reference.index.name = "event_datetime"
        self.simulations = {}
        self.results = {}

    @staticmethod
    def normalize_dataframe(data, value_name="values"):
        """
        Convert a DataFrame to the canonical ForecastPerformance long format.

        The canonical format is a single-column DataFrame whose row index is a
        ``pd.MultiIndex`` containing the relevant subset of:

        * ``production_datetime`` — time at which the forecast was issued
        * ``event_datetime``      — time to which the forecast refers
        * ``leadtime``            — forecast horizon (``event_datetime - production_datetime``)
        * ``non_exceedance``      — non-exceedance probability level (probabilistic)
        * ``ensemble_member``     — ensemble member identifier (ensemble)

        The following naming conventions are accepted and normalised:

        .. code-block:: text

            probability / prob / quantile / percentile  →  non_exceedance
            ensemble / ensemble_number / member         →  ensemble_member
            lead_time / lead                            →  leadtime
            production / production_date                →  production_datetime
            event / event_date                          →  event_datetime

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Input data.  Accepted layouts:

            * **Wide** — row index is ``production_datetime`` (or another
              datetime level), columns are a ``pd.MultiIndex`` with levels
              ``leadtime``, ``non_exceedance``, or ``ensemble_member``.
            * **Long** — already a single-column DataFrame with a
              ``pd.MultiIndex`` row index.

        value_name : str, optional
            Name for the single output column.  Default ``"values"``.

        Returns
        -------
        pd.DataFrame
            Single-column DataFrame named *value_name* with a canonical
            ``pd.MultiIndex``.  If exactly two of
            ``production_datetime``, ``event_datetime``, ``leadtime`` are
            present the third is derived automatically.

        Raises
        ------
        TypeError
            If *data* is neither a ``pd.Series`` nor a ``pd.DataFrame``.

        Examples
        --------
        Convert a wide probabilistic DataFrame::

            wide = pd.DataFrame(
                ...,
                columns=pd.MultiIndex.from_product(
                    [leadtimes, quantiles], names=["leadtime", "probability"]
                ),
            )
            wide.index.name = "production_datetime"
            long = ForecastPerformance.normalize_dataframe(wide)
            # MultiIndex: (production_datetime, event_datetime, leadtime, non_exceedance)
        """
        if isinstance(data, pd.Series):
            data = data.to_frame(name=value_name)
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "data must be a pandas Series or DataFrame, "
                "got %s." % type(data).__name__
            )

        data = data.copy()

        # Normalise index level names.
        if isinstance(data.index, pd.MultiIndex):
            data.index.names = [_normalise_name(n) for n in data.index.names]
        else:
            data.index.name = _normalise_name(data.index.name)

        # Normalise column level names.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.set_names(
                [_normalise_name(n) for n in data.columns.names]
            )

        # Stack column levels into the row index to produce long format.
        if isinstance(data.columns, pd.MultiIndex):
            stacked = data.stack(list(range(data.columns.nlevels)), future_stack=True)
            data = stacked.to_frame(name=value_name) if isinstance(stacked, pd.Series) else stacked
            data.columns = [value_name]
        elif data.shape[1] == 1:
            data.columns = [value_name]
        else:
            stacked = data.stack(future_stack=True)
            data = stacked.to_frame(name=value_name) if isinstance(stacked, pd.Series) else stacked
            data.columns = [value_name]

        # Re-normalise index level names introduced by stacking.
        if isinstance(data.index, pd.MultiIndex):
            data.index.names = [_normalise_name(n) for n in data.index.names]
        else:
            data.index.name = _normalise_name(data.index.name)

        # Promote any unnamed datetime level to production_datetime (or event_datetime).
        if isinstance(data.index, pd.MultiIndex):
            new_names = list(data.index.names)
            for i, n in enumerate(new_names):
                if n is None and str(data.index.get_level_values(i).dtype).startswith("datetime"):
                    new_names[i] = (
                        "production_datetime"
                        if "production_datetime" not in new_names
                        else "event_datetime"
                    )
            data.index.names = new_names
        elif data.index.name is None and str(data.index.dtype).startswith("datetime"):
            data.index.name = "production_datetime"

        # Derive the missing one of production_datetime / event_datetime / leadtime.
        names = list(data.index.names) if isinstance(data.index, pd.MultiIndex) else [data.index.name]
        has_prod = "production_datetime" in names
        has_event = "event_datetime" in names
        has_lt = "leadtime" in names

        def _lvl(df, name):
            return df.index.get_level_values(name) if isinstance(df.index, pd.MultiIndex) else df.index

        if has_prod and has_lt and not has_event:
            event = _lvl(data, "production_datetime") + _lvl(data, "leadtime")
            data = data.assign(event_datetime=event).set_index("event_datetime", append=True)
        elif has_event and has_lt and not has_prod:
            prod = _lvl(data, "event_datetime") - _lvl(data, "leadtime")
            data = data.assign(production_datetime=prod).set_index("production_datetime", append=True)
        elif has_prod and has_event and not has_lt:
            lt = _lvl(data, "event_datetime") - _lvl(data, "production_datetime")
            data = data.assign(leadtime=lt).set_index("leadtime", append=True)

        return data

    @_validate_simulation_input
    def add(self, data, name, leadtime=pd.Timedelta("0D"), sort=True):
        """
        Register a simulation.

        Parameters
        ----------
        data : pd.DataFrame
            Simulation data in canonical long format **or** in wide format
            (datetime row index, ``pd.MultiIndex`` columns with levels
            ``leadtime``, ``non_exceedance`` or ``ensemble_member``).
            :meth:`normalize_dataframe` is called automatically.
        name : str
            Identifier for this simulation.
        leadtime : timedelta, optional
            Default leadtime used when none is encoded in *data*.
        sort : bool
            Sort quantile forecasts to ensure non-decreasing order across
            ``non_exceedance`` levels.
        """
        long = self.normalize_dataframe(data)

        # ---- ensure leadtime is in the index --------------------------------
        idx_names = list(long.index.names) if isinstance(long.index, pd.MultiIndex) else [long.index.name]

        if "leadtime" not in idx_names:
            if not isinstance(long.index, pd.MultiIndex):
                if long.index.name not in ("production_datetime", "event_datetime"):
                    long.index.name = "production_datetime"
            long = long.assign(leadtime=leadtime).set_index("leadtime", append=True)
            idx_names = list(long.index.names)

        # ---- ensure both datetime levels are present ------------------------
        has_prod = "production_datetime" in idx_names
        has_event = "event_datetime" in idx_names

        if has_prod and not has_event:
            prod = long.index.get_level_values("production_datetime")
            lt_vals = long.index.get_level_values("leadtime")
            long = long.assign(event_datetime=prod + lt_vals).set_index(
                "event_datetime", append=True
            )
        elif has_event and not has_prod:
            event = long.index.get_level_values("event_datetime")
            lt_vals = long.index.get_level_values("leadtime")
            long = long.assign(production_datetime=event - lt_vals).set_index(
                "production_datetime", append=True
            )

        idx_names = list(long.index.names)

        # ---- detect simulation type -----------------------------------------
        if "non_exceedance" in idx_names:
            simulation_type = "probabilistic"
        elif "ensemble_member" in idx_names:
            simulation_type = "ensemble"
        else:
            simulation_type = "simple"

        # ---- type-specific processing ---------------------------------------
        if simulation_type == "ensemble":
            try:
                ens_idx = long.index.names.index("ensemble_member")
                long.index = long.index.set_levels(
                    [int(v) for v in long.index.levels[ens_idx]],
                    level="ensemble_member",
                )
            except Exception:
                pass
            long = long.sort_index()
            n = long.index.get_level_values("ensemble_member").nunique()
            probabilities = np.cumsum(np.array([1.0 / n] * n))

        elif simulation_type == "probabilistic":
            ne_vals = long.index.get_level_values("non_exceedance")
            if len(ne_vals) > 0 and isinstance(ne_vals[0], str) and "%" in str(ne_vals[0]):
                ne_idx = long.index.names.index("non_exceedance")
                float_levels = np.sort(
                    [float(str(v).replace("%", "")) / 100 for v in long.index.levels[ne_idx]]
                )
                long.index = long.index.set_levels(float_levels, level="non_exceedance")
            long = long.sort_index()
            probabilities = np.sort(
                long.index.get_level_values("non_exceedance").unique().astype(float)
            )
            if sort:
                group_levels = ["production_datetime", "event_datetime", "leadtime"]
                sorted_vals = (
                    long.groupby(level=group_levels)["values"]
                    .transform(lambda x: np.sort(x.values))
                )
                long = long.copy()
                long["values"] = sorted_vals

        else:
            long = long.sort_index()
            probabilities = [1]

        # ---- canonical level order ------------------------------------------
        current = list(long.index.names)
        ordered = [n for n in _LEVEL_ORDER if n in current] + [
            n for n in current if n not in _LEVEL_ORDER
        ]
        if ordered != current:
            long = long.reorder_levels(ordered)
            long = long.sort_index()

        # ---- extract leadtimes ----------------------------------------------
        try:
            leadtimes_list = sorted(long.index.get_level_values("leadtime").unique())
        except TypeError:
            leadtimes_list = list(long.index.get_level_values("leadtime").unique())

        self.simulations[name] = {
            "data": long,
            "leadtimes": leadtimes_list,
            "simulationType": simulation_type,
            "probabilities": probabilities,
        }
        self.results[name] = {}

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def names(self):
        """Return a list of registered simulation names."""
        return list(self.simulations.keys())

    def leadtimes(self):
        """Return a boolean table of leadtimes per simulation."""
        names = self.names()
        leadtime_idx = set()
        for s0 in names:
            leadtime_idx.update(self.simulations[s0]["leadtimes"])
        try:
            idx = pd.TimedeltaIndex(np.unique(list(leadtime_idx)), name="leadtime")
        except Exception:
            idx = pd.Index(np.unique(list(leadtime_idx)), name="leadtime")
        table = pd.DataFrame(False, index=idx, columns=names)
        for s0 in names:
            table.loc[table.index.isin(self.simulations[s0]["leadtimes"]), s0] = True
        return table

    # ------------------------------------------------------------------
    # Cropping
    # ------------------------------------------------------------------

    def crop_event_dates(self, start=None, end=None):
        """Discard data outside [start, end] based on **event** dates."""
        if start:
            self.reference = self.reference.loc[self.reference.index >= start]
        if end:
            self.reference = self.reference.loc[self.reference.index <= end]
        for k0, s0 in self.simulations.items():
            data = s0["data"]
            ev = data.index.get_level_values("event_datetime")
            if start:
                data = data.loc[ev >= start]
            if end:
                data = data.loc[ev <= end]
            self.simulations[k0]["data"] = data

    def crop_production_dates(self, start=None, end=None):
        """Discard data outside [start, end] based on **production** dates."""
        if start:
            self.reference = self.reference.loc[self.reference.index >= start]
        if end:
            self.reference = self.reference.loc[self.reference.index <= end]
        for k0, s0 in self.simulations.items():
            data = s0["data"]
            pd_ = data.index.get_level_values("production_datetime")
            mask = np.ones(len(data), dtype=bool)
            if start:
                mask &= pd_ >= start
            if end:
                mask &= pd_ <= end
            self.simulations[k0]["data"] = data.loc[mask]

    # ------------------------------------------------------------------
    # Bias / scale corrections
    # ------------------------------------------------------------------

    def adjust_mean(self, name):
        """Shift ensemble members so the ensemble mean matches the reference mean."""
        if self.simulations[name]["simulationType"] != "ensemble":
            raise Exception("The mean can only be adjusted in ensemble simulations.")
        ref_mean = np.nanmean(self.reference.values.ravel())
        for lt in self.simulations[name]["leadtimes"]:
            mask = self.simulations[name]["data"].index.get_level_values("leadtime") == lt
            ens_mean = np.nanmean(self.simulations[name]["data"].loc[mask, "values"])
            if not np.isnan(ens_mean):
                self.simulations[name]["data"].loc[mask, "values"] -= ens_mean - ref_mean

    def adjust_scale(self, name):
        """Scale ensemble members so the ensemble mean matches the reference mean."""
        if self.simulations[name]["simulationType"] != "ensemble":
            raise Exception("The scale can only be adjusted in ensemble simulations.")
        ref_mean = np.nanmean(self.reference.values.ravel())
        for lt in self.simulations[name]["leadtimes"]:
            mask = self.simulations[name]["data"].index.get_level_values("leadtime") == lt
            ens_mean = np.nanmean(self.simulations[name]["data"].loc[mask, "values"])
            if ens_mean != 0 and not np.isnan(ens_mean):
                self.simulations[name]["data"].loc[mask, "values"] *= ref_mean / ens_mean

    # ------------------------------------------------------------------
    # Reference / baseline forecasts
    # ------------------------------------------------------------------

    def get_persistence(self, leadtimes):
        """
        Return a persistence forecast as a canonical long-format DataFrame.

        The persistence value at each (production_datetime, leadtime) is the
        reference observation at that production time.
        """
        leadtimes = self._convert_leadtimes(leadtimes)
        rows = []
        for lt in leadtimes:
            ref = self.reference.copy()
            prod_dt = ref.index
            event_dt = prod_dt + lt
            mi = pd.MultiIndex.from_arrays(
                [prod_dt, event_dt, [lt] * len(prod_dt)],
                names=["production_datetime", "event_datetime", "leadtime"],
            )
            rows.append(
                pd.DataFrame({"values": ref.iloc[:, 0].values}, index=mi)
            )
        return pd.concat(rows).sort_index()

    def get_climatology(
        self,
        multiplicative=False,
        leadtimes=None,
        rolling_window=61,
        non_exceedance=None,
        coefficients=9,
        minimum=-np.inf,
        maximum=np.inf,
    ):
        """
        Build a climatological probabilistic forecast using a Fourier-fitted
        seasonal cycle plus empirical residual quantiles.

        Returns a canonical long-format DataFrame with index levels
        ``(production_datetime, event_datetime, leadtime, non_exceedance)``.
        """
        if leadtimes is None:
            leadtimes = [pd.Timedelta("0D")]
        if non_exceedance is None:
            non_exceedance = [
                0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                0.6, 0.7, 0.8, 0.9, 0.95, 0.99,
            ]

        reference_ = self.reference.dropna()
        time_ = reference_.index.map(
            lambda x: (x - pd.Timestamp(x.year, 1, 1)).total_seconds()
        ).values.ravel()

        def fourier(x, time):
            n = int((len(x) - 3) / 2)
            year = 2 * np.pi / 365.25 / 86400
            day = 2 * np.pi / 86400
            series = x[0]
            for i in range(1, n + 1):
                series += x[i] * np.sin(i * time * year) + x[i + n] * np.cos(
                    i * time * year
                )
            series += x[-2] * np.sin(time * day) + x[-1] * np.cos(time * day)
            return series

        def error(x, time, reference):
            return np.square(fourier(x, time) - reference).mean()

        x = minimize(
            error,
            x0=[1.0] * (coefficients + 2),
            args=(time_, reference_.values.ravel()),
        )

        cycle = self.reference.copy()
        time = self.reference.index.map(
            lambda x: (x - pd.Timestamp(x.year, 1, 1)).total_seconds()
        ).values.ravel()
        cycle.iloc[:, 0] = fourier(x.x, time)

        difference = self.reference - cycle
        difference.loc[:, "time"] = time

        percentiles = (
            difference.groupby("time").quantile(q=non_exceedance).unstack(-1)
        )

        to_prepend = percentiles.iloc[-rolling_window:, :]
        to_append = percentiles.iloc[:rolling_window, :]
        percentiles_ = pd.concat((to_prepend, percentiles, to_append), axis=0)
        percentiles_ = percentiles_.rolling(
            window=rolling_window, min_periods=1, center=True
        ).mean()
        percentiles = percentiles_.iloc[rolling_window:-rolling_window, :]

        climatology = (
            pd.concat([cycle] * percentiles.shape[1], axis=1)
            + percentiles.reindex(time).values
        )

        clim_parts = []
        for l0 in leadtimes:
            tmp = climatology.copy()
            tmp.index = tmp.index - l0
            tmp.columns = pd.MultiIndex.from_product(
                [[l0], non_exceedance], names=["leadtime", "non_exceedance"]
            )
            clim_parts.append(tmp)
        wide = pd.concat(clim_parts, axis=1)
        wide[wide < minimum] = minimum
        wide[wide > maximum] = maximum
        wide.index.name = "production_datetime"

        # Stack to canonical long format.
        long = wide.stack(["leadtime", "non_exceedance"], future_stack=True).to_frame(
            name="values"
        )
        prod = long.index.get_level_values("production_datetime")
        lt = long.index.get_level_values("leadtime")
        long = long.assign(event_datetime=prod + lt).set_index(
            "event_datetime", append=True
        )
        long = long.reorder_levels(
            ["production_datetime", "event_datetime", "leadtime", "non_exceedance"]
        )
        return long.sort_index()

    # ------------------------------------------------------------------
    # Probabilistic metrics
    # ------------------------------------------------------------------

    def _resolve_probabilistic_metric(self, metric):
        """Map metric callables and aliases to a canonical metric key."""
        key = (metric.__name__ if callable(metric) else str(metric)).lower().strip()
        aliases = {
            "quantile_loss": "quantile_loss",
            "crps": "crps",
            "fair_crps": "fair_crps",
            "faircrps": "fair_crps",
            "reliability": "reliability",
            "resolution": "resolution",
            "resolution_relative": "resolution_relative",
            "brier_score": "brier_score",
            "briers": "brier_score",
            "fair_brier_score": "fair_brier_score",
            "fairbriers": "fair_brier_score",
            "fair_crps_skill_score": "fair_crps_skill_score",
            "faircrpss": "fair_crps_skill_score",
            "fair_brier_skill_score": "fair_brier_skill_score",
            "fairbrierss": "fair_brier_skill_score",
        }
        if key not in aliases:
            raise Exception("Unsupported probabilistic metric: %s" % key)
        return aliases[key]

    def _prepare_probabilistic_inputs(self, name, leadtime, months=None):
        """Return aligned (simulations, probabilities, targets, index, type)."""
        if hasattr(leadtime, "__iter__") and not isinstance(leadtime, str):
            raise Exception(
                "The probabilistic wrapper expects a single leadtime, not an iterable."
            )
        data = self._slice_leadtime(name, leadtime)
        tmp = pd.concat((data, self.reference), axis=1).dropna(how="any")
        if months is not None:
            tmp = tmp.loc[tmp.index.month.isin(months)]
        targets = tmp.loc[:, ["Reference"]].values
        simulations = tmp.drop("Reference", axis=1).values
        probabilities = np.asarray(self.simulations[name]["probabilities"])
        simulation_type = self.simulations[name]["simulationType"]
        return simulations, probabilities, targets, tmp.index, simulation_type

    def probabilistic(self, metric, name, leadtime, months=None, metric_kwargs=None):
        """
        Apply a probabilistic metric to one leadtime slice.

        Parameters
        ----------
        metric : callable or str
            Probabilistic metric identifier.
        name : str
            Simulation name.
        leadtime : timedelta
            Single leadtime to evaluate.
        months : iterable[int] or None
            Optional subset of months to include.
        metric_kwargs : dict or None
            Extra keyword arguments (e.g. ``threshold``, ``reference``).
        """
        metric_kwargs = {} if metric_kwargs is None else dict(metric_kwargs)
        metric_name = self._resolve_probabilistic_metric(metric)
        simulations, probabilities, targets, index, simulation_type = (
            self._prepare_probabilistic_inputs(name, leadtime, months=months)
        )

        if metric_name == "quantile_loss":
            if simulation_type != "probabilistic":
                raise Exception(
                    '"quantile_loss" can only be applied to probabilistic data.'
                )
            return _prob.quantile_loss(simulations, probabilities, targets)

        if metric_name == "crps":
            if simulation_type == "probabilistic" and (
                probabilities[0] != 0 or probabilities[-1] != 1
            ):
                warnings.warn(
                    "Boundaries of the probabilistic forecast are incomplete (not [0,1])."
                )
            p_values = self._p_values(name, leadtime).loc[index, :].values
            return _prob.crps(
                simulations, probabilities, targets,
                simulation_type=simulation_type, p_values=p_values,
            )

        if metric_name == "fair_crps":
            p_values = self._p_values(name, leadtime).loc[index, :].values
            return _prob.fair_crps(
                simulations, probabilities, targets,
                simulation_type=simulation_type, p_values=p_values,
            )

        if metric_name == "reliability":
            p_values = self._p_values(name, leadtime).loc[index, :].values
            return _prob.reliability(p_values)

        if metric_name in ("resolution", "resolution_relative"):
            relative = bool(metric_kwargs.get("relative", False))
            if metric_name == "resolution_relative":
                relative = True
            return _prob.resolution(
                simulations, probabilities,
                simulation_type=simulation_type, relative=relative,
            )

        if metric_name in ("brier_score", "fair_brier_score"):
            threshold = metric_kwargs.get("threshold")
            if threshold is None:
                raise Exception("Please provide threshold in metric_kwargs.")
            pv = self._p_values(name, leadtime, threshold).loc[index, :]
            if metric_name == "brier_score" or simulation_type != "ensemble":
                brier = _prob.brier_score(pv.values, targets, threshold)
            else:
                brier = _prob.fair_brier_score(
                    pv.values, targets, threshold, n_members=probabilities.size
                )
            if bool(metric_kwargs.get("return_p_values", False)):
                return brier, pv
            return brier

        if metric_name == "fair_crps_skill_score":
            reference = metric_kwargs.get("reference")
            if reference is None:
                raise Exception("Please provide reference in metric_kwargs.")
            reference_leadtime = metric_kwargs.get("reference_leadtime")
            if reference_leadtime is None:
                if len(self.simulations[reference]["leadtimes"]) > 1:
                    raise Exception("The reference cannot have more than one leadtime.")
                reference_leadtime = self.simulations[reference]["leadtimes"][0]
            return _prob.fair_crps_skill_score(
                self.probabilistic(_prob.fair_crps, name, leadtime, months=months),
                self.probabilistic(_prob.fair_crps, reference, reference_leadtime, months=months),
            )

        if metric_name == "fair_brier_skill_score":
            reference = metric_kwargs.get("reference")
            threshold = metric_kwargs.get("threshold")
            if reference is None:
                raise Exception("Please provide reference in metric_kwargs.")
            if threshold is None:
                raise Exception("Please provide threshold in metric_kwargs.")
            reference_leadtime = metric_kwargs.get("reference_leadtime")
            if reference_leadtime is None:
                if len(self.simulations[reference]["leadtimes"]) > 1:
                    raise Exception("The reference cannot have more than one leadtime.")
                reference_leadtime = self.simulations[reference]["leadtimes"][0]
            return _prob.fair_brier_skill_score(
                self.probabilistic(
                    _prob.fair_brier_score, name, leadtime, months=months,
                    metric_kwargs={"threshold": threshold},
                ),
                self.probabilistic(
                    _prob.fair_brier_score, reference, reference_leadtime, months=months,
                    metric_kwargs={"threshold": threshold},
                ),
            )

        raise Exception("Unknown probabilistic metric: %s" % metric_name)

    # ------------------------------------------------------------------
    # Deterministic wrapper
    # ------------------------------------------------------------------

    def get_expected(self, name, leadtime=None):
        """
        Return the expected (mean) forecast as a canonical long-format DataFrame.

        Returns a single-column DataFrame with a ``(production_datetime,
        event_datetime, leadtime)`` MultiIndex.  If *leadtime* is ``None``
        all registered leadtimes are included; raises ``ValueError`` when
        there is more than one registered leadtime.
        """
        if leadtime is None and len(self.simulations[name]["leadtimes"]) > 1:
            raise ValueError(
                "leadtime=None is ambiguous for '%s' which has %d leadtimes; "
                "pass an explicit leadtime." % (name, len(self.simulations[name]["leadtimes"]))
            )
        simulation_type = self.simulations[name]["simulationType"]
        data = self.simulations[name]["data"]

        if leadtime is not None:
            lt_vals = data.index.get_level_values("leadtime")
            mask = lt_vals == leadtime
            if not mask.any():
                mask = lt_vals == (
                    pd.Timedelta("0H")
                    if leadtime == pd.DateOffset(days=0)
                    else pd.DateOffset(days=0)
                )
            data = data[mask]

        if simulation_type == "simple":
            return data.rename(columns={"values": name})

        group_levels = ["production_datetime", "event_datetime", "leadtime"]

        if simulation_type == "ensemble":
            return (
                data.groupby(level=group_levels)["values"]
                .mean()
                .to_frame(name=name)
            )

        # probabilistic: expected value via trapezoidal integration over CDF
        probs = self.simulations[name]["probabilities"]
        weights = np.diff(np.r_[0.0, np.asarray(probs, dtype=float), 1.0])

        def _ev(group):
            vals = group.values
            midpoints = np.r_[vals[0], vals[:-1] + np.diff(vals) / 2.0, vals[-1]]
            return float(np.dot(weights, midpoints))

        return (
            data.groupby(level=group_levels)["values"]
            .apply(_ev)
            .to_frame(name=name)
        )

    def get_expected_prediction(self, name):
        """Return expected values for all leadtimes as a single long-format DataFrame."""
        parts = [
            self.get_expected(name, lt)
            for lt in self.simulations[name]["leadtimes"]
        ]
        return pd.concat(parts).sort_index()

    def deterministic(self, metric, name, leadtime=None):
        """
        Apply a deterministic metric function to the expected forecast.

        Parameters
        ----------
        metric : callable
            A function ``(simulations, targets) -> scalar``.
        name : str
        leadtime : timedelta or None
        """
        expected = self.get_expected(name=name, leadtime=leadtime)
        ev_idx = pd.DatetimeIndex(expected.index.get_level_values("event_datetime"))
        exp_series = pd.Series(expected.iloc[:, 0].values, index=ev_idx)
        tmp = pd.concat([exp_series, self.reference.iloc[:, 0]], axis=1).dropna()
        return metric(tmp.iloc[:, 0].values, tmp.iloc[:, 1].values)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def qq_plot(self, name, leadtimes=None, plot=True, ax=None):
        """
        Return a DataFrame of p-values vs Uniform(0,1) quantiles.

        Parameters
        ----------
        name : str
            Simulation name.
        leadtimes : iterable or None
            Leadtimes to include.  Defaults to all registered leadtimes.
        plot : bool
            Whether to render the Q-Q plot.  Default ``True``.
        ax : matplotlib Axes or None

        Returns
        -------
        pd.DataFrame
            Columns ``uniform``, ``p_values``, ``leadtime``.
        """
        if leadtimes is None:
            leadtimes = self.simulations[name]["leadtimes"]
        else:
            leadtimes = self._convert_leadtimes(leadtimes)

        frames = []
        for lt in leadtimes:
            pv = self._p_values(name, lt)
            sorted_pv = np.sort(pv.values.ravel())
            uniform = np.linspace(0, 1, sorted_pv.size)
            frames.append(
                pd.DataFrame({"uniform": uniform, "p_values": sorted_pv, "leadtime": lt})
            )
        result = pd.concat(frames, ignore_index=True)

        if plot:
            if ax is None:
                _, ax = plt.subplots()
            for lt, grp in result.groupby("leadtime"):
                ax.plot(grp["uniform"].values, grp["p_values"].values, label=lt)
            ax.plot([0, 1], [0, 1], ":k")
            ax.set_xlim([-0.01, 1.01])
            ax.set_ylim([-0.01, 1.01])
            ax.legend()
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Theoretical quantile of U[0,1]")
            ax.set_ylabel("Quantile of the observed p-values")
            ax.set_title(name)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _slice_leadtime(self, name, leadtime):
        """
        Return wide-format data for a single leadtime, indexed by event_datetime.

        Columns are ``non_exceedance`` or ``ensemble_member`` levels for
        probabilistic/ensemble simulations, or a single unnamed column for
        simple simulations.
        """
        if leadtime is None and len(self.simulations[name]["leadtimes"]) > 1:
            raise ValueError(
                "leadtime=None is ambiguous for '%s' which has %d leadtimes; "
                "pass an explicit leadtime." % (name, len(self.simulations[name]["leadtimes"]))
            )
        data = self.simulations[name]["data"]

        if leadtime is not None:
            lt_vals = data.index.get_level_values("leadtime")
            mask = lt_vals == leadtime
            if not mask.any():
                alt = (
                    pd.Timedelta("0H")
                    if leadtime == pd.DateOffset(days=0)
                    else pd.DateOffset(days=0)
                )
                mask = lt_vals == alt
            data = data[mask]

        extra = next(
            (lvl for lvl in ("non_exceedance", "ensemble_member") if lvl in data.index.names),
            None,
        )

        if extra:
            result = data["values"].unstack(extra)
        else:
            result = data[["values"]]

        # Drop all index levels except event_datetime.
        if isinstance(result.index, pd.MultiIndex):
            drop = [n for n in result.index.names if n != "event_datetime"]
            if drop:
                result = result.droplevel(drop)

        return result

    def _convert_leadtimes(self, leadtimes):
        """Normalise a leadtime argument into a list."""
        if leadtimes is None:
            return [None]
        if hasattr(leadtimes, "__iter__") and not isinstance(leadtimes, str):
            return list(leadtimes)
        return [leadtimes]

    @storedResults()
    def _p_values(self, name, leadtime=None, threshold=None):
        """
        Compute PIT p-values.

        Cached per (name, leadtime); bypassed when a threshold is given.
        """
        data = self._slice_leadtime(name, leadtime)
        probabilities = np.asarray(self.simulations[name]["probabilities"])
        simulation_type = self.simulations[name]["simulationType"]

        tmp = pd.concat((data, self.reference), axis=1).dropna(how="any")
        targets = tmp.loc[:, ["Reference"]].copy()
        if threshold is not None:
            targets.loc[:, :] = threshold
        simulations = tmp.drop("Reference", axis=1)

        if simulation_type == "ensemble":
            p_vals = _prob.p_values_ensemble(
                simulations.values, probabilities, targets.values
            )
        elif simulation_type == "probabilistic":
            p_vals = _prob.p_values_probabilistic(
                simulations.values, probabilities, targets.values
            )
        elif simulation_type == "simple":
            pv = (simulations.values < targets.values).astype(float)
            pv[simulations.values == targets.values] = 0.5
            p_vals = pv.ravel()
        else:
            raise Exception(
                "Not possible to compute p-values for %s simulations." % simulation_type
            )

        return pd.DataFrame({"p-values": p_vals}, index=simulations.index)
