"""
ForecastPerformance — main class for evaluating deterministic and
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
        leadtimes = data.columns.get_level_values("Leadtime")
        data_ = []
        for i0 in range(data.shape[1]):
            tmp = data.iloc[:, [i0]].copy()
            tmp.index = tmp.index + leadtimes[i0]
            data_.append(tmp)
        data_ = pd.concat(data_, axis=1)
        data_.index.name = "Event date"
        return data_

    @staticmethod
    def event_to_production_dates(data):
        """Shift an event-date-indexed DataFrame back to production dates."""
        leadtimes = data.columns.get_level_values("Leadtime")
        data_ = []
        for i0 in range(data.shape[1]):
            tmp = data.iloc[:, [i0]].copy()
            tmp.index = tmp.index - leadtimes[i0]
            data_.append(tmp)
        data_ = pd.concat(data_, axis=1)
        data_.index.name = "Production date"
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
            with levels ``Leadtime``, ``Probability``, or ``Ensemble member``.
        name : str
            Identifier for this simulation.
        leadtime : timedelta, optional
            Default leadtime when none is encoded in the columns.
        sort : bool
            Whether to sort quantile forecasts to ensure non-decreasing order.
        """
        data = data.copy(deep=True)
        # Set the index name early so stack/unstack("Production date") works
        # regardless of whether the caller named the index.
        data.index.name = "Production date"
        try:
            data.columns = data.columns.remove_unused_levels()
        except AttributeError:
            pass

        if not isinstance(data.columns, pd.MultiIndex):
            data.columns = pd.MultiIndex.from_frame(data.columns.to_frame())

        if "Leadtime" not in data.columns.names:
            leadtimes = [leadtime]
        else:
            leadtimes = data.columns.get_level_values("Leadtime").unique()

        if (
            "Probability" not in data.columns.names
            and "Ensemble member" not in data.columns.names
        ) or data.shape[1] / len(leadtimes) == 1:
            simulation_type = "simple"
            probabilities = [1]

        elif "Ensemble member" in data.columns.names:
            simulation_type = "ensemble"
            n = data.columns.get_level_values("Ensemble member").unique().shape[0]
            probabilities = np.cumsum(np.array([1.0 / n] * n))
            try:
                ens_inx = list(data.columns.names).index("Ensemble member")
                tmp = data.columns.levels[ens_inx]
                tmp = [int(i) for i in tmp]
                data.columns = data.columns.set_levels(tmp, level=ens_inx)
            except Exception:
                pass
            data.sort_index(axis=1, level="Ensemble member", inplace=True)

        elif "Probability" in data.columns.names:
            simulation_type = "probabilistic"
            try:
                prob_inx = list(data.columns.names).index("Probability")
                probabilities = data.columns.levels[prob_inx]
                if isinstance(probabilities[0], str) and "%" in probabilities[0]:
                    probabilities = probabilities.str.replace("%", "").map(
                        lambda x: float(x) / 100
                    )
                probabilities = np.sort(probabilities.astype(float))
                data.columns = data.columns.set_levels(probabilities, level=prob_inx)
            except Exception:
                pass
            data.sort_index(axis=1, level=["Probability"], inplace=True)

            if sort:
                tmp = data.stack("Probability").unstack("Production date")
                tmp = pd.DataFrame(
                    np.sort(tmp.values, axis=0), index=tmp.index, columns=tmp.columns
                )
                data = tmp.stack("Production date").unstack("Probability")
        else:
            raise Exception("Problem identifying the type of simulation…")

        if data.shape[1] != len(leadtimes) * len(probabilities):
            raise Exception(
                "The shape of the input data %s does not agree with the number of "
                "leadtimes (%u) and probability bands/ensemble members (%u)"
                % (str(data.shape), len(leadtimes), len(probabilities))
            )

        data.index.name = "Production date"

        if len(data.columns.levels) == 1:
            data = pd.concat([data], keys=[0], names=["Dummy"], axis=1)

        leadtimes = self._convert_leadtimes(leadtimes)

        if "Leadtime" not in data.columns.names:
            tmp = data.columns.to_frame()
            if len(leadtimes) > 1:
                raise Exception("Please check leadtimes…")
            tmp.loc[:, "Leadtime"] = leadtimes[0]
            data.columns = pd.MultiIndex.from_frame(tmp)

        lt_inx = list(data.columns.names).index("Leadtime")
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

        table = pd.DataFrame(False, index=pd.TimedeltaIndex(np.unique(list(leadtime_idx)), name='Leadtime'), columns=names)
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
        persistence.columns = pd.Index(leadtimes, name="Leadtime")
        persistence.index.names = ["Production date"]
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
            [leadtimes, probabilities], names=["Leadtime", "Probability"]
        )

        climatology[climatology < minimum] = minimum
        climatology[climatology > maximum] = maximum
        climatology.index.names = ["Production date"]
        return climatology

    # ------------------------------------------------------------------
    # Probabilistic metrics
    # ------------------------------------------------------------------

    def quantile_loss(self, name, leadtimes):
        """
        Mean quantile (pinball) loss.

        Only applicable to probabilistic simulations.
        """
        if self.simulations[name]["simulationType"] != "probabilistic":
            raise Exception(
                'The "quantile_loss" metric can only be applied to probabilistic data.'
            )

        probabilities = np.asarray(self.simulations[name]["probabilities"])
        leadtimes = self._convert_leadtimes(leadtimes)
        quantile_losses = []
        for leadtime in leadtimes:
            data = self._getLeadtime(self.simulations[name]["data"], leadtime)
            tmp = pd.concat((data, self.reference), axis=1).dropna(how="any")
            targets = tmp.loc[:, ["Reference"]].values
            simulations = tmp.drop("Reference", axis=1).values
            quantile_losses.append(_prob.quantile_loss(simulations, probabilities, targets))

        if len(quantile_losses) > 1:
            return pd.DataFrame(
                quantile_losses,
                index=pd.Index(leadtimes, name="Leadtime"),
                columns=pd.Index([name], name="Quantile loss"),
            ).transpose()
        return quantile_losses[0]

    @storedResults()
    def CRPS(self, name, leadtime, lowThreshold=None, highThreshold=None, months=None):
        """Continuous ranked probability score (CRPS)."""
        if highThreshold is not None:
            raise Exception("Computation for high threshold not implemented.")
        if lowThreshold is not None:
            raise Exception("Computation for low threshold not implemented.")

        data = self._getLeadtime(self.simulations[name]["data"], leadtime)
        probabilities = np.asarray(self.simulations[name]["probabilities"])
        simulation_type = self.simulations[name]["simulationType"]

        tmp = pd.concat((data, self.reference), axis=1).dropna(how="any")
        targets = tmp.loc[:, ["Reference"]]
        simulations = tmp.drop("Reference", axis=1)

        p_values = self._p_values(name, leadtime)
        p_values_aligned = p_values.loc[simulations.index, :].values  # (n, 1)

        if simulation_type == "ensemble":
            integral_vals = _prob.crps_ensemble_integral(
                simulations.values, probabilities, targets.values, p_values_aligned
            )
            integral = pd.DataFrame({"integral": integral_vals}, index=simulations.index)

        elif simulation_type == "probabilistic":
            if probabilities[0] != 0 or probabilities[-1] != 1:
                warnings.warn(
                    "Boundaries of the probabilistic forecast are incomplete (not [0,1])."
                )
            probs_c = probabilities.copy()
            sims_c = simulations.values.copy()
            if probabilities[0] != 0:
                probs_c = np.r_[0.0, probs_c]
                sims_c = np.c_[sims_c[:, 0], sims_c]
            if probabilities[-1] != 1:
                probs_c = np.r_[probs_c, 1.0]
                sims_c = np.c_[sims_c, sims_c[:, -1]]

            integral_vals = _prob.crps_probabilistic_integral(
                sims_c, probs_c, targets.values, p_values_aligned
            )
            integral = pd.DataFrame({"integral": integral_vals}, index=simulations.index)

        elif simulation_type == "simple":
            vals = np.abs(simulations.values - targets.values).ravel()
            integral = pd.DataFrame({"integral": vals}, index=simulations.index)

        else:
            raise Exception(
                "Not possible to compute CRPS for %s simulations." % simulation_type
            )

        if months is None:
            valid = np.ones(len(integral), dtype=bool)
        else:
            valid = integral.index.month.isin(months)

        result = float(np.mean(integral.loc[valid, "integral"].values))
        return result

    @storedResults()
    def fairCRPS(self, name, leadtime, lowThreshold=None, highThreshold=None, months=None):
        """Fair CRPS (removes ensemble-size bias for ensemble forecasts)."""
        if highThreshold is not None:
            raise Exception("Computation for high threshold not implemented.")
        if lowThreshold is not None:
            raise Exception("Computation for low threshold not implemented.")

        crps = self.CRPS(
            name,
            leadtime=leadtime,
            lowThreshold=lowThreshold,
            highThreshold=highThreshold,
            months=months,
        )

        if self.simulations[name]["simulationType"] != "ensemble":
            return crps

        probabilities = np.asarray(self.simulations[name]["probabilities"])
        data = self._getLeadtime(self.simulations[name]["data"], leadtime)
        tmp = pd.concat((data, self.reference), axis=1).dropna(how="any")
        simulations = tmp.drop("Reference", axis=1)

        spread_per_sample = _prob.fair_crps_ensemble_spread(
            simulations.values, probabilities
        )
        spread_df = pd.DataFrame(
            {"integral": spread_per_sample}, index=simulations.index
        )

        if months is None:
            valid = np.ones(len(spread_df), dtype=bool)
        else:
            valid = spread_df.index.month.isin(months)

        return crps - float(np.mean(spread_df.loc[valid, "integral"].values))

    def reliability(self, name, leadtimes):
        """
        Reliability (alpha score) after Renard et al. (2010).

        Values closer to 1 indicate better calibration.
        """
        leadtimes = self._convert_leadtimes(leadtimes)
        alphas = []
        for leadtime in leadtimes:
            p_values = self._p_values(name, leadtime)
            sorted_pv = np.sort(p_values.values.ravel())
            uniform = np.linspace(0, 1, sorted_pv.size)
            alpha_prime = np.abs(sorted_pv - uniform).mean()
            alphas.append(1.0 - 2.0 * alpha_prime)

        if len(alphas) > 1:
            return pd.DataFrame(
                alphas,
                index=pd.Index(leadtimes, name="Leadtime"),
                columns=pd.Index([name], name="Reliability"),
            ).transpose()
        return alphas[0]

    def resolution(self, name, leadtimes, relative=False):
        """
        Resolution (sharpness) after Renard et al. (2010).

        Parameters
        ----------
        relative : bool
            If ``True`` returns mean/std (relative); otherwise 1/std (absolute).
        """
        leadtimes = self._convert_leadtimes(leadtimes)
        pis = []
        metric = "Resolution"
        for leadtime in leadtimes:
            data = self._getLeadtime(self.simulations[name]["data"], leadtime)
            sim_type = self.simulations[name]["simulationType"]
            if sim_type == "probabilistic":
                # Numerical integration: probabilities = interval widths between
                # [0, q1, q2, ..., qN, 1]; intermediate = midpoints + boundary values.
                # Both arrays have n_quantiles+1 elements → shapes match.
                probs = np.diff(
                    np.unique(np.r_[0, self.simulations[name]["probabilities"], 1])
                )
                probs = np.tile(probs, (data.shape[0], 1))
                intermediate = np.c_[
                    data.values[:, [0]],
                    data.values[:, :-1] + data.diff(axis=1).values[:, 1:] / 2,
                    data.values[:, [-1]],
                ]
                means = np.sum(probs * intermediate, axis=1)
                means_ = np.tile(means, (probs.shape[1], 1)).T
                stds = np.sqrt(
                    np.sum(np.square(intermediate - means_) * probs, axis=1)
                )
                statistics = pd.DataFrame({"Mean": means, "Std": stds}, index=data.index)
            elif sim_type == "ensemble":
                # For equal-weight ensembles use the direct sample mean/std.
                # The midpoint formula produces n_members+1 columns but weight
                # arrays have n_members columns → shape mismatch.
                means = data.values.mean(axis=1)
                stds = data.values.std(axis=1, ddof=0)
                statistics = pd.DataFrame({"Mean": means, "Std": stds}, index=data.index)
            else:
                pis.append(np.inf)
                continue

            if relative:
                pi = float(np.mean(statistics["Mean"] / statistics["Std"]))
                metric = "Resolution (relative)"
            else:
                pi = float(np.mean(1.0 / statistics["Std"]))
                metric = "Resolution (absolute)"
            pis.append(pi)

        if len(pis) > 1:
            return pd.DataFrame(
                pis,
                index=pd.Index(leadtimes, name="Leadtime"),
                columns=pd.Index([name], name=metric),
            ).transpose()
        return pis[0]

    def resolution_relative(self, name, leadtimes):
        """Relative resolution (mean/std)."""
        return self.resolution(name, leadtimes, relative=True)

    def BrierS(self, name, threshold, leadtime, returnPValues=False, months=None):
        """Brier score for threshold exceedance."""
        data = self._getLeadtime(self.simulations[name]["data"], leadtime)
        tmp = pd.concat((data, self.reference), axis=1).dropna(how="any")
        targets = tmp.loc[:, ["Reference"]]

        p_values = self._p_values(name, leadtime, threshold)
        p_values_ = p_values.loc[targets.index, :]
        heaviside = np.heaviside(threshold - targets.values, 1)
        brier = np.square(p_values_.values - heaviside)

        if months is None:
            valid = np.ones(len(p_values_), dtype=bool)
        else:
            valid = p_values_.index.month.isin(months)

        brier_ = brier[valid]
        if returnPValues:
            return float(np.mean(brier_)), p_values_.loc[valid]
        return float(np.mean(brier_))

    def fairBrierS(self, name, threshold, leadtime, months=None):
        """Fair Brier score (removes ensemble-size bias for ensemble forecasts)."""
        brier, p_values_ = self.BrierS(
            name, threshold, leadtime=leadtime, returnPValues=True, months=months
        )
        if self.simulations[name]["simulationType"] == "ensemble":
            n_members = np.asarray(self.simulations[name]["probabilities"]).size
            return brier - float(
                np.average(p_values_.values * (1.0 - p_values_.values) / (n_members - 1))
            )
        return brier

    def fairCRPSS(self, name, reference, leadtime):
        """
        Fair CRPS skill score against a reference simulation.

        The reference must have exactly one leadtime.
        """
        if len(self.simulations[reference]["leadtimes"]) > 1:
            raise Exception("The reference cannot have more than one leadtime.")

        ref_leadtime = self.simulations[reference]["leadtimes"][0]  # fix
        fair_crps = self.fairCRPS(name, leadtime=leadtime)
        fair_crps_ref = self.fairCRPS(reference, leadtime=ref_leadtime)  # fix
        return 1.0 - fair_crps / fair_crps_ref

    def fairBrierSS(self, name, reference, threshold, leadtime):
        """
        Fair Brier skill score against a reference simulation.

        The reference must have exactly one leadtime.
        """
        if len(self.simulations[reference]["leadtimes"]) > 1:
            raise Exception("The reference cannot have more than one leadtime.")

        ref_leadtime = self.simulations[reference]["leadtimes"][0]            # fix
        fair_brier = self.fairBrierS(name, threshold, leadtime=leadtime)
        fair_brier_ref = self.fairBrierS(reference, threshold, leadtime=ref_leadtime)  # fix
        return 1.0 - fair_brier / fair_brier_ref

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
                    [(name, leadtime)], names=["Name", "Leadtime"]
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
            production_dates = seaborn_df["Production date"].unique()

        seaborn_df = seaborn_df.loc[seaborn_df["Production date"].isin(production_dates), :]
        key = seaborn_df.columns[3]
        seaborn_df = seaborn_df.sort_values(by=[key, "Production date", "Leadtime"])
        seaborn_df = seaborn_df.loc[
            (seaborn_df["Production date"] >= date_from)
            & (seaborn_df["Production date"] <= date_to), :
        ]
        seaborn_df = seaborn_df.sort_values(by="Production date")

        if estimator:
            ax = sns.lineplot(
                x="Event date", y="Variable", hue="Production date",
                errorbar=errorbar, data=seaborn_df, palette=palette,
                ax=ax, estimator=estimator,
            )
        else:
            ax = sns.lineplot(
                x="Event date", y="Variable", hue="Production date",
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
            (seaborn_df["Event date"] >= date_from)
            & (seaborn_df["Event date"] <= date_to), :
        ]

        seaborn_ = seaborn_df.loc[seaborn_df["Leadtime"].map(lambda x: x in leadtimes), :]
        if seaborn_.shape[0] == 0 and len(leadtimes) == 1:
            if leadtimes[0] == pd.DateOffset(days=0):
                seaborn_ = seaborn_df.loc[seaborn_df["Leadtime"] == pd.Timedelta("0H"), :]
            elif leadtimes[0] == pd.Timedelta("0H"):
                seaborn_ = seaborn_df.loc[seaborn_df["Leadtime"] == pd.DateOffset(days=0), :]

        seaborn_df = seaborn_.copy()
        seaborn_df.loc[:, "Leadtime"] = seaborn_df["Leadtime"].map(str)
        seaborn_df = seaborn_df.sort_values(by="Leadtime")

        ax = sns.lineplot(
            x="Event date", y="Variable", hue="Leadtime",
            errorbar=errorbar, data=seaborn_df, palette=palette, ax=ax, zorder=3,
        )
        if reference:
            tmp_years = pd.DatetimeIndex(seaborn_df["Event date"]).year
            tmp = self.reference.loc[
                (self.reference.index.year >= tmp_years.min())
                & (self.reference.index.year <= tmp_years.max()), :
            ]
            tmp.columns = ["Reference"]
            tmp.plot(ax=ax, style="--k", linewidth=1, zorder=2)
        return ax

    def QQPlot(self, name, leadtimes=None, ax=None):
        """Q–Q plot of p-values against the Uniform(0,1) distribution."""
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

        seaborn_df.columns = ["Variable"]
        seaborn_df = seaborn_df.reset_index()
        if "Leadtime" not in seaborn_df.columns:
            seaborn_df.loc[:, "Leadtime"] = self.simulations[name]["leadtimes"][0]
        seaborn_df.loc[:, "Event date"] = (
            seaborn_df["Production date"] + seaborn_df["Leadtime"]
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
            data_ = data.loc[:, data.columns.get_level_values("Leadtime") == leadtime]
            if data_.shape[1] == 0:
                if leadtime == pd.DateOffset(days=0):
                    data_ = data.loc[
                        :, data.columns.get_level_values("Leadtime") == pd.Timedelta("0H")
                    ]
                elif leadtime == pd.Timedelta("0H"):
                    data_ = data.loc[
                        :, data.columns.get_level_values("Leadtime") == pd.DateOffset(days=0)
                    ]
            data = data_.copy()          # fix: copy before mutating index
            data.index = data.index + leadtime
            data.index.name = "Event date"

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
        lt_idx = list(self.simulations[name]["data"].columns.names).index("Leadtime")
        index = tuple([slice(None)] * lt_idx + [leadtime])
        self.simulations[name]["data"].loc[:, index] = data.values
