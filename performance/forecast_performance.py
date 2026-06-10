"""
ForecastPerformance - main class for evaluating deterministic and
probabilistic weather / hydrological forecasts.
"""

import warnings
import datetime as dt

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from .decorators import storedResults
from .metrics import deterministic as _det
from .metrics import probabilistic as _prob


class ForecastPerformance:
    """
    Evaluate deterministic, ensemble and probabilistic forecasts against a
    reference (observation) time series.

    Parameters
    ----------
    reference : pd.Series or pd.DataFrame
        Observed values indexed by date.  A single-column DataFrame or a
        Series are both accepted.
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

    def __init__(self, reference):
        self.reference = reference
        if not isinstance(self.reference, pd.DataFrame):
            self.reference = self.reference.to_frame()
        self.reference.columns = ["Reference"]
        self.simulations = {}
        self.results = {}

    @staticmethod
    def production_to_event_dates(data):
        """Shift a production-date-indexed DataFrame to event dates."""
        leadtimes = data.columns.get_level_values("leadtime")
        data_ = []
        for i0 in range(data.shape[1]):
            tmp = data.iloc[:, [i0]].copy()
            tmp.index = tmp.index + leadtimes[i0]
            data_.append(tmp)
        data_ = pd.concat(data_, axis=1)
        data_.index.name = "event_datetime"
        return data_

    @staticmethod
    def event_to_production_dates(data):
        """Shift an event-date-indexed DataFrame back to production dates."""
        leadtimes = data.columns.get_level_values("leadtime")
        data_ = []
        for i0 in range(data.shape[1]):
            tmp = data.iloc[:, [i0]].copy()
            tmp.index = tmp.index - leadtimes[i0]
            data_.append(tmp)
        data_ = pd.concat(data_, axis=1)
        data_.index.name = "production_datetime"
        return data_

    def add_by_production_date(self, *args, **kwargs):
        """Add a simulation indexed by production date."""
        self._add(*args, **kwargs)

    def add_by_event_date(self, data, *args, **kwargs):
        """Add a simulation indexed by event date (converted internally)."""
        data_ = self.event_to_production_dates(data)
        self._add(data_, *args, **kwargs)

    def _add(self, data, name, leadtime=pd.Timedelta("0D"), sort=True):
        """
        Register a simulation.

        Parameters
        ----------
        data : pd.DataFrame
            Production-date-indexed DataFrame.  Columns may carry a MultiIndex
            with levels ``leadtime``, ``non_exceedance``, or ``Ensemble member``.
        name : str
            Identifier for this simulation.
        leadtime : timedelta, optional
            Default leadtime when none is encoded in the columns.
        sort : bool
            Whether to sort quantile forecasts to ensure non-decreasing order.
        """
        data = data.copy(deep=True)
        # Set the index name early so stack/unstack("production_datetime") works
        # regardless of whether the caller named the index.
        data.index.name = "production_datetime"
        try:
            data.columns = data.columns.remove_unused_levels()
        except AttributeError:
            pass

        if not isinstance(data.columns, pd.MultiIndex):
            data.columns = pd.MultiIndex.from_frame(data.columns.to_frame())

        # Normalise common column-level aliases (e.g. "Leadtime", "Probability").
        canonical = {
            "leadtime": "leadtime",
            "non_exceedance": "non_exceedance",
            "probability": "non_exceedance",
            "ensemble_member": "ensemble_member",
        }
        normalised_names = []
        for level_name in data.columns.names:
            if level_name is None:
                normalised_names.append(level_name)
                continue
            key = str(level_name).strip().lower().replace(" ", "_")
            normalised_names.append(canonical.get(key, key))
        data.columns = data.columns.set_names(normalised_names)

        if "leadtime" not in data.columns.names:
            leadtimes = [leadtime]
        else:
            leadtimes = data.columns.get_level_values("leadtime").unique()

        if (
            "non_exceedance" not in data.columns.names
            and "ensemble_member" not in data.columns.names
        ) or data.shape[1] / len(leadtimes) == 1:
            simulation_type = "simple"
            probabilities = [1]

        elif "ensemble_member" in data.columns.names:
            simulation_type = "ensemble"
            n = data.columns.get_level_values("ensemble_member").unique().shape[0]
            probabilities = np.cumsum(np.array([1.0 / n] * n))
            try:
                ens_inx = list(data.columns.names).index("ensemble_member")
                tmp = data.columns.levels[ens_inx]
                tmp = [int(i) for i in tmp]
                data.columns = data.columns.set_levels(tmp, level=ens_inx)
            except Exception:
                pass
            data.sort_index(axis=1, level="ensemble_member", inplace=True)

        elif "non_exceedance" in data.columns.names:
            simulation_type = "probabilistic"
            try:
                prob_inx = list(data.columns.names).index("non_exceedance")
                probabilities = data.columns.levels[prob_inx]
                if isinstance(probabilities[0], str) and "%" in probabilities[0]:
                    probabilities = probabilities.str.replace("%", "").map(
                        lambda x: float(x) / 100
                    )
                probabilities = np.sort(probabilities.astype(float))
                data.columns = data.columns.set_levels(probabilities, level=prob_inx)
            except Exception:
                pass
            data.sort_index(axis=1, level=["non_exceedance"], inplace=True)

            if sort:
                tmp = data.stack("non_exceedance").unstack("production_datetime")
                tmp = pd.DataFrame(
                    np.sort(tmp.values, axis=0), index=tmp.index, columns=tmp.columns
                )
                data = tmp.stack("production_datetime").unstack("non_exceedance")
        else:
            raise Exception("Problem identifying the type of simulation...")

        if data.shape[1] != len(leadtimes) * len(probabilities):
            raise Exception(
                "The shape of the input data %s does not agree with the number of "
                "leadtimes (%u) and probability bands/ensemble members (%u)"
                % (str(data.shape), len(leadtimes), len(probabilities))
            )

        data.index.name = "production_datetime"

        if len(data.columns.levels) == 1:
            data = pd.concat([data], keys=[0], names=["Dummy"], axis=1)

        leadtimes = self._convert_leadtimes(leadtimes)

        if "leadtime" not in data.columns.names:
            tmp = data.columns.to_frame()
            if len(leadtimes) > 1:
                raise Exception("Please check leadtimes...")
            tmp.loc[:, "leadtime"] = leadtimes[0]
            data.columns = pd.MultiIndex.from_frame(tmp)

        lt_inx = list(data.columns.names).index("leadtime")
        tmp = data.columns.levels[lt_inx]
        data.columns = data.columns.set_levels(
            self._convert_leadtimes(tmp), level=lt_inx
        )

        self.simulations[name] = {
            "data": data,
            "leadtimes": leadtimes,
            "simulationType": simulation_type,
            "probabilities": probabilities,
        }
        self.results[name] = {}

    def names(self):
        """Return a list of registered simulation names."""
        return list(self.simulations.keys())
    
    def leadtimes(self):
        """Return a table of registered leadtimes."""
        names = self.names()
        leadtimes = []
        leadtime_idx = set()
        for s0 in names:
            leadtimes.append(self.simulations[s0]['leadtimes'])
            leadtime_idx.update(leadtimes[-1])

        table = pd.DataFrame(False, index=pd.TimedeltaIndex(np.unique(list(leadtime_idx)), name='leadtime'), columns=names)
        for i0, s0 in enumerate(names):
            table.loc[table.index.isin(leadtimes[i0]), s0] = True
        return table

    # ------------------------------------------------------------------
    # Cropping
    # ------------------------------------------------------------------

    def crop_event_dates(self, start=None, end=None):
        """Discard data outside [start, end] based on **event** dates."""
        if start:
            self.reference = self.reference.loc[self.reference.index >= start, :]
        if end:
            self.reference = self.reference.loc[self.reference.index <= end, :]

        for k0, s0 in self.simulations.items():
            by_event = self.production_to_event_dates(s0["data"])
            if start:
                by_event = by_event.loc[by_event.index >= start, :]
            if end:
                by_event = by_event.loc[by_event.index <= end, :]
            self.simulations[k0]["data"] = self.event_to_production_dates(by_event)

    def crop_production_dates(self, start=None, end=None):
        """Discard data outside [start, end] based on **production** dates.

        Bug fix: original code referenced undefined ``k0`` inside the loop.
        """
        if start:
            self.reference = self.reference.loc[self.reference.index >= start, :]
        if end:
            self.reference = self.reference.loc[self.reference.index <= end, :]

        for k0, s0 in self.simulations.items():           # fix: k0 (was _)
            by_production = s0["data"].copy()
            if start:
                by_production = by_production.loc[by_production.index >= start, :]
            if end:
                by_production = by_production.loc[by_production.index <= end, :]
            self.simulations[k0]["data"] = by_production

    # ------------------------------------------------------------------
    # Bias / scale corrections
    # ------------------------------------------------------------------

    def adjustMean(self, name):
        """Shift ensemble members so the ensemble mean matches the reference mean."""
        if self.simulations[name]["simulationType"] != "ensemble":
            raise Exception("The mean can only be adjusted in ensemble simulations.")

        ref_mean = np.nanmean(self.reference.values.ravel())
        for leadtime in self.simulations[name]["leadtimes"]:
            data = self._getLeadtime(self.simulations[name]["data"], leadtime)
            ens_mean = np.nanmean(data.values.ravel())
            if not np.isnan(ens_mean):
                data -= ens_mean - ref_mean
            self._setLeadtime(data, name, leadtime)

    def adjustScale(self, name):
        """Scale ensemble members so the ensemble mean matches the reference mean."""
        if self.simulations[name]["simulationType"] != "ensemble":
            raise Exception("The scale can only be adjusted in ensemble simulations.")

        ref_mean = np.nanmean(self.reference.values.ravel())
        for leadtime in self.simulations[name]["leadtimes"]:
            data = self._getLeadtime(self.simulations[name]["data"], leadtime)
            ens_mean = np.nanmean(data.values.ravel())
            if ens_mean != 0 and not np.isnan(ens_mean):
                data *= ref_mean / ens_mean
            self._setLeadtime(data, name, leadtime)

    # ------------------------------------------------------------------
    # Reference / baseline forecasts
    # ------------------------------------------------------------------

    def get_persistence(self, leadtimes):
        """Return a persistence forecast for the given leadtimes."""
        persistence = pd.concat([self.reference] * len(leadtimes), axis=1)
        persistence.columns = pd.Index(leadtimes, name="leadtime")
        persistence.index.names = ["production_datetime"]
        return persistence

    def get_climatology(
        self,
        multiplicative=False,
        leadtimes=None,
        rolling_window=61,
        probabilities=None,
        coefficients=9,
        minimum=-np.inf,
        maximum=np.inf,
    ):
        """
        Build a climatological probabilistic forecast using a Fourier-fitted
        seasonal cycle plus empirical residual quantiles.
        """
        if leadtimes is None:
            leadtimes = [pd.Timedelta("0D")]
        if probabilities is None:
            probabilities = [
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

        x = minimize(error, x0=[1.0] * (coefficients + 2), args=(time_, reference_.values.ravel()))

        cycle = self.reference.copy()
        time = self.reference.index.map(
            lambda x: (x - pd.Timestamp(x.year, 1, 1)).total_seconds()
        ).values.ravel()
        cycle.iloc[:, 0] = fourier(x.x, time)

        difference = self.reference - cycle
        difference.loc[:, "time"] = time

        percentiles = (
            difference.groupby("time")
            .quantile(q=probabilities)
            .unstack(-1)
        )

        to_prepend = percentiles.iloc[-rolling_window:, :]
        to_append = percentiles.iloc[:rolling_window, :]
        percentiles_ = pd.concat((to_prepend, percentiles, to_append), axis=0)

        # Fix: removed deprecated axis=0 argument (removed in pandas 2.2)
        percentiles_ = percentiles_.rolling(
            window=rolling_window, min_periods=1, center=True
        ).mean()
        percentiles = percentiles_.iloc[rolling_window:-rolling_window, :]

        climatology = (
            pd.concat([cycle] * percentiles.shape[1], axis=1)
            + percentiles.reindex(time).values
        )

        climatology_ = []
        for l0 in leadtimes:
            tmp = climatology.copy()
            tmp.index = tmp.index - l0
            climatology_.append(tmp)
        climatology = pd.concat(climatology_, axis=1)
        climatology.columns = pd.MultiIndex.from_product(
            [leadtimes, probabilities], names=["leadtime", "non_exceedance"]
        )

        climatology[climatology < minimum] = minimum
        climatology[climatology > maximum] = maximum
        climatology.index.names = ["production_datetime"]
        return climatology

    # ------------------------------------------------------------------
    # Probabilistic metrics
    # ------------------------------------------------------------------

    def _resolve_probabilistic_metric(self, metric):
        """Map metric callables and aliases to a canonical metric key."""
        if callable(metric):
            key = metric.__name__
        else:
            key = str(metric)

        key = key.lower().strip()
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
        """Prepare aligned simulations, probabilities and targets for one leadtime."""
        if hasattr(leadtime, "__iter__") and not isinstance(leadtime, str):
            raise Exception(
                "The probabilistic wrapper expects a single leadtime, not an iterable."
            )

        data = self._getLeadtime(self.simulations[name]["data"], leadtime)
        tmp = pd.concat((data, self.reference), axis=1).dropna(how="any")

        if months is not None:
            tmp = tmp.loc[tmp.index.month.isin(months), :]

        targets = tmp.loc[:, ["Reference"]].values
        simulations = tmp.drop("Reference", axis=1).values
        probabilities = np.asarray(self.simulations[name]["probabilities"])
        simulation_type = self.simulations[name]["simulationType"]
        return simulations, probabilities, targets, tmp.index, simulation_type

    def probabilistic(self, metric, name, leadtime, months=None, metric_kwargs=None):
        """
        Apply a probabilistic metric function to one leadtime forecast slice.

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
            Extra arguments for metrics needing additional inputs,
            for example threshold or reference.
        """
        metric_kwargs = {} if metric_kwargs is None else dict(metric_kwargs)
        metric_name = self._resolve_probabilistic_metric(metric)
        simulations, probabilities, targets, index, simulation_type = (
            self._prepare_probabilistic_inputs(name, leadtime, months=months)
        )

        if metric_name == "quantile_loss":
            if simulation_type != "probabilistic":
                raise Exception(
                    'The "quantile_loss" metric can only be applied to probabilistic data.'
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
                simulations,
                probabilities,
                targets,
                simulation_type=simulation_type,
                p_values=p_values,
            )

        if metric_name == "fair_crps":
            p_values = self._p_values(name, leadtime).loc[index, :].values
            return _prob.fair_crps(
                simulations,
                probabilities,
                targets,
                simulation_type=simulation_type,
                p_values=p_values,
            )

        if metric_name == "reliability":
            p_values = self._p_values(name, leadtime).loc[index, :].values
            return _prob.reliability(p_values)

        if metric_name in ("resolution", "resolution_relative"):
            relative = bool(metric_kwargs.get("relative", False))
            if metric_name == "resolution_relative":
                relative = True
            return _prob.resolution(
                simulations,
                probabilities,
                simulation_type=simulation_type,
                relative=relative,
            )

        if metric_name in ("brier_score", "fair_brier_score"):
            threshold = metric_kwargs.get("threshold")
            if threshold is None:
                raise Exception("Please provide threshold in metric_kwargs.")

            p_values_df = self._p_values(name, leadtime, threshold).loc[index, :]
            return_p_values = bool(metric_kwargs.get("return_p_values", False))

            if metric_name == "brier_score" or simulation_type != "ensemble":
                score = _prob.brier_score(p_values_df.values, targets, threshold)
            else:
                score = _prob.fair_brier_score(
                    p_values_df.values,
                    targets,
                    threshold,
                    n_members=probabilities.size,
                )

            if return_p_values:
                return score, p_values_df
            return score

        if metric_name == "fair_crps_skill_score":
            reference = metric_kwargs.get("reference")
            if reference is None:
                raise Exception("Please provide reference in metric_kwargs.")

            reference_leadtime = metric_kwargs.get("reference_leadtime")
            if reference_leadtime is None:
                if len(self.simulations[reference]["leadtimes"]) > 1:
                    raise Exception("The reference cannot have more than one leadtime.")
                reference_leadtime = self.simulations[reference]["leadtimes"][0]

            score = self.probabilistic(
                _prob.fair_crps,
                name,
                leadtime,
                months=months,
            )
            score_ref = self.probabilistic(
                _prob.fair_crps,
                reference,
                reference_leadtime,
                months=months,
            )
            return _prob.fair_crps_skill_score(score, score_ref)

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

            score = self.probabilistic(
                _prob.fair_brier_score,
                name,
                leadtime,
                months=months,
                metric_kwargs={"threshold": threshold},
            )
            score_ref = self.probabilistic(
                _prob.fair_brier_score,
                reference,
                reference_leadtime,
                months=months,
                metric_kwargs={"threshold": threshold},
            )
            return _prob.fair_brier_skill_score(score, score_ref)

        raise Exception("Unknown probabilistic metric: %s" % metric_name)

    # ------------------------------------------------------------------
    # Deterministic wrapper
    # ------------------------------------------------------------------

    def get_expected_value(self, name, leadtime=None):
        """Return the expected (mean) forecast as a Series/DataFrame."""
        simulation_type = self.simulations[name]["simulationType"]
        data = self._getLeadtime(self.simulations[name]["data"], leadtime)

        if simulation_type == "simple":
            return data

        if simulation_type == "ensemble":
            return data.mean(axis=1)

        if simulation_type == "probabilistic":
            probs = np.diff(np.unique(np.r_[0, data.columns.astype(float), 1]))
            probs = np.tile(probs, (data.shape[0], 1))
            intermediate = np.c_[
                data.values[:, [0]],
                data.values[:, :-1] + data.diff(axis=1).values[:, 1:] / 2,
                data.values[:, [-1]],
            ]
            means = np.sum(probs * intermediate, axis=1)
            return pd.DataFrame(
                means,
                index=data.index,
                columns=pd.MultiIndex.from_tuples(
                    [(name, leadtime)], names=["Name", "leadtime"]
                ),
            )
        return None

    def get_expected_prediction(self, name):
        """Return expected values for all leadtimes as a single DataFrame."""
        parts = [
            self.get_expected_value(name, lt)
            for lt in self.simulations[name]["leadtimes"]
        ]
        return pd.concat(parts, axis=1).sort_index(axis=1)

    def deterministic(self, metric, name, leadtime=None):
        """
        Apply a deterministic metric function to the expected forecast.

        Parameters
        ----------
        metric : callable
            A function ``(data, reference) -> scalar``, e.g.
            :func:`~performance.metrics.rmse`.
        name : str
        leadtime : timedelta or None
        """
        tmp = pd.concat(
            (self.get_expected_value(name=name, leadtime=leadtime), self.reference),
            axis=1,
        ).dropna(how="any")
        targets = tmp.loc[:, ["Reference"]]
        simulations = tmp.drop("Reference", axis=1)
        return metric(simulations.values.ravel(), targets.values.ravel())

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(
        self,
        name,
        production_dates=None,
        reference=True,
        start_markers=True,
        ax=None,
        legend=False,
        cumulative=False,
        palette="plasma",
        errorbar=("pi", 100),
        date_from=dt.datetime(1900, 1, 1),
        date_to=dt.datetime(2100, 1, 1),
        estimator=np.mean,
    ):
        """Plot forecast envelopes for selected production dates."""
        seaborn_df = self._getSeaborn(name)

        if production_dates is None:
            production_dates = seaborn_df["production_datetime"].unique()

        seaborn_df = seaborn_df.loc[seaborn_df["production_datetime"].isin(production_dates), :]
        key = seaborn_df.columns[3]
        seaborn_df = seaborn_df.sort_values(by=[key, "production_datetime", "leadtime"])
        seaborn_df = seaborn_df.loc[
            (seaborn_df["production_datetime"] >= date_from)
            & (seaborn_df["production_datetime"] <= date_to), :
        ]
        seaborn_df = seaborn_df.sort_values(by="production_datetime")

        if estimator:
            ax = sns.lineplot(
                x="event_datetime", y="variable", hue="production_datetime",
                errorbar=errorbar, data=seaborn_df, palette=palette,
                ax=ax, estimator=estimator,
            )
        else:
            ax = sns.lineplot(
                x="event_datetime", y="variable", hue="production_datetime",
                units=key, errorbar=errorbar, data=seaborn_df, palette=palette,
                ax=ax, estimator=estimator,
            )

        reference_data = self.reference.loc[
            (self.reference.index >= date_from) & (self.reference.index <= date_to), :
        ]

        # Fix: always fetch lines so start_markers can use them regardless of
        # whether cumulative is True or False
        lines = ax.get_lines()

        if cumulative:
            reference_data = reference_data.cumsum()
            for line in lines:
                x_data, y_data = line.get_data()
                if x_data.shape[0] > 0:
                    anchor = reference_data.loc[
                        pd.Timestamp("1970-1-1") + pd.Timedelta(days=x_data[0])
                    ].values
                    new_y = anchor + np.cumsum(y_data) - y_data[0]
                    line.set_data(x_data, new_y)
            ax.relim()
            ax.autoscale_view()
            plt.draw()

        if start_markers:
            for line in lines:
                x_data, y_data = line.get_data()
                color = line.get_color()
                if x_data.shape[0] > 0:
                    plt.scatter(x_data[0], y_data[0], color=color, s=100)

        if not legend:
            ax.get_legend().remove()

        if reference:
            reference_data.columns = ["Reference"]
            reference_data.plot(ax=ax, style=":k")

        return ax

    def leadtimePlot(
        self,
        name,
        leadtimes=None,
        reference=True,
        ax=None,
        palette="plasma",
        errorbar=("pi", 100),
        date_from=dt.datetime(1900, 1, 1),
        date_to=dt.datetime(2100, 1, 1),
    ):
        """Plot forecast envelopes grouped by leadtime."""
        if leadtimes is None:
            leadtimes = [pd.Timedelta("0H")]

        seaborn_df = self._getSeaborn(name)
        seaborn_df = seaborn_df.loc[
            (seaborn_df["event_datetime"] >= date_from)
            & (seaborn_df["event_datetime"] <= date_to), :
        ]

        seaborn_ = seaborn_df.loc[seaborn_df["leadtime"].map(lambda x: x in leadtimes), :]
        if seaborn_.shape[0] == 0 and len(leadtimes) == 1:
            if leadtimes[0] == pd.DateOffset(days=0):
                seaborn_ = seaborn_df.loc[seaborn_df["leadtime"] == pd.Timedelta("0H"), :]
            elif leadtimes[0] == pd.Timedelta("0H"):
                seaborn_ = seaborn_df.loc[seaborn_df["leadtime"] == pd.DateOffset(days=0), :]

        seaborn_df = seaborn_.copy()
        seaborn_df.loc[:, "leadtime"] = seaborn_df["leadtime"].map(str)
        seaborn_df = seaborn_df.sort_values(by="leadtime")

        ax = sns.lineplot(
            x="event_datetime", y="variable", hue="leadtime",
            errorbar=errorbar, data=seaborn_df, palette=palette, ax=ax, zorder=3,
        )
        if reference:
            tmp_years = pd.DatetimeIndex(seaborn_df["event_datetime"]).year
            tmp = self.reference.loc[
                (self.reference.index.year >= tmp_years.min())
                & (self.reference.index.year <= tmp_years.max()), :
            ]
            tmp.columns = ["Reference"]
            tmp.plot(ax=ax, style="--k", linewidth=1, zorder=2)
        return ax

    def QQPlot(self, name, leadtimes=None, ax=None):
        """Q-Q plot of p-values against the Uniform(0,1) distribution."""
        leadtimes = self._convert_leadtimes(leadtimes)
        for leadtime in leadtimes:
            p_values = self._p_values(name, leadtime)
            sorted_pv = np.sort(p_values.values.ravel())
            uniform = np.linspace(0, 1, sorted_pv.size)
            if ax is None:
                plt.figure()
                ax = plt.gca()
            else:
                plt.sca(ax)
            plt.plot(uniform, sorted_pv, label=leadtime)

        plt.plot([0, 1], [0, 1], ":k")
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.legend()
        ax.set_aspect("equal", adjustable="box")
        plt.xlabel("Theoretical quantile of U[0,1]")
        plt.ylabel("Quantile of the observed p-values")
        plt.title(name)
        return ax

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def getSimulationNames(self):
        """Return the names of all registered simulations."""
        return self.simulations.keys()

    def getSimulationLeadtimes(self, name):
        """Return the leadtimes for a given simulation."""
        return self.simulations[name]["leadtimes"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _getSeaborn(self, name):
        """Return a long-format DataFrame suitable for seaborn plotting."""
        if "seaborn" in self.results[name]:
            return self.results[name]["seaborn"]

        seaborn_df = self.simulations[name]["data"]
        try:
            seaborn_df = seaborn_df.stack(
                tuple(range(len(seaborn_df.columns.levels)))
            ).to_frame()
        except Exception:
            try:
                seaborn_df = seaborn_df.stack().to_frame()
            except Exception:
                pass

        seaborn_df.columns = ["variable"]
        seaborn_df = seaborn_df.reset_index()
        if "leadtime" not in seaborn_df.columns:
            seaborn_df.loc[:, "leadtime"] = self.simulations[name]["leadtimes"][0]
        seaborn_df.loc[:, "event_datetime"] = (
            seaborn_df["production_datetime"] + seaborn_df["leadtime"]
        )
        self.results[name]["seaborn"] = seaborn_df
        return seaborn_df

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
        Compute probability-integral-transform p-values.

        Cached per (name, leadtime); bypassed when threshold is given.
        """
        data = self._getLeadtime(
            self.simulations[name]["data"], leadtime, drop_redundant=False
        )
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

    def _getLeadtime(self, data, leadtime=None, drop_redundant=True):
        """
        Return a DataFrame slice for a specific leadtime.

        Fix: work on a copy so the original index is never mutated in-place.
        """
        if leadtime is not None:
            data_ = data.loc[:, data.columns.get_level_values("leadtime") == leadtime]
            if data_.shape[1] == 0:
                if leadtime == pd.DateOffset(days=0):
                    data_ = data.loc[
                        :, data.columns.get_level_values("leadtime") == pd.Timedelta("0H")
                    ]
                elif leadtime == pd.Timedelta("0H"):
                    data_ = data.loc[
                        :, data.columns.get_level_values("leadtime") == pd.DateOffset(days=0)
                    ]
            data = data_.copy()          # fix: copy before mutating index
            data.index = data.index + leadtime
            data.index.name = "event_datetime"

        if drop_redundant and isinstance(data.columns, pd.MultiIndex):
            for c0 in range(len(data.columns.levels) - 1, -1, -1):
                if (
                    data.columns.get_level_values(c0).unique().size == 1
                    and data.shape[1] > 1
                ):
                    data.columns = data.columns.droplevel(c0)

        return data

    def _setLeadtime(self, data, name, leadtime):
        """Overwrite the stored data for a specific leadtime."""
        lt_idx = list(self.simulations[name]["data"].columns.names).index("leadtime")
        index = tuple([slice(None)] * lt_idx + [leadtime])
        self.simulations[name]["data"].loc[:, index] = data.values
