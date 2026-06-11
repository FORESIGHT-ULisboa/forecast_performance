<p align="center">
  <img src="notebooks/foresight.png" alt="FORESIGHT" width="320">
</p>

# Forecast Performance

**`forecast_performance`** is a Python library created by [FORESIGHT - Forecasting and Optimization for Resilient Environmental Systems through Investigation with Groundbreaking Hydrological Tools](https://foresight.tecnico.ulisboa.pt/) for evaluating the skill of
deterministic and probabilistic forecasting models.  It provides a single
unified interface — `ForecastPerformance` — that handles point, quantile, and
ensemble forecasts and exposes a rich set of metrics and visualisation tools.

> **New here? Start with the notebooks** in [`notebooks/`](notebooks/), each a
> self-contained, runnable walkthrough:
> [`00_visualize`](notebooks/00_visualize.ipynb) (Plotly forecast plots) ·
> [`01_benchmarks`](notebooks/01_benchmarks.ipynb) (persistence & climatology
> baselines) · [`02_deterministic`](notebooks/02_deterministic.ipynb) ·
> [`03_ensemble`](notebooks/03_ensemble.ipynb) ·
> [`04_probabilistic`](notebooks/04_probabilistic.ipynb). Run them under the
> `forecast_performance` kernel (see [Installation](#installation)).

---

## Features

| Category | Metrics / tools |
|---|---|
| **Deterministic** | RMSE, MAE, MSE, Bias, Relative bias, Pearson r, Spearman r, NSE, KGE, KGE' |
| **Probabilistic** | CRPS, Fair CRPS, Fair CRPS skill score, Brier score, Fair Brier score, Fair Brier skill score, Quantile loss |
| **Reliability** | PIT / Q-Q plot, Reliability index, Resolution (sharpness) |
| **Baselines** | `get_persistence`, `get_climatology` |
| **Post-processing** | `adjust_mean`, `adjust_scale` |
| **Visualisation** | `qq_plot`, `plotly_forecasting` helpers |
| **Utilities** | `Results` accumulator, `storedResults` caching decorator |

---

## Installation

### 1  Create a conda environment

```bat
conda create -n forecast_performance python=3.11
conda activate forecast_performance
```

### 2  Install the package and all dependencies

From the repository root, run:

```bat
pip install -e ".[dev]"
```

### 3  Register the Jupyter kernel

So the notebooks pick up the right environment:

```bat
python -m ipykernel install --user --name forecast_performance --display-name "forecast_performance"
```

### 4  Open the notebooks

Open them directly in VS Code and select the **forecast_performance** kernel in
the top-right kernel picker, or run:

```bat
jupyter lab notebooks\
```

---

## Core concepts

### Canonical long format

Internally every forecast is stored as a **single-column** `DataFrame` whose
row index is a `MultiIndex` drawn from these levels:

| Level | Meaning |
|---|---|
| `production_datetime` | time the forecast was issued |
| `event_datetime` | time the forecast refers to |
| `leadtime` | `event_datetime - production_datetime` |
| `non_exceedance` | quantile (non-exceedance probability) level — *probabilistic* |
| `ensemble_member` | ensemble member id — *ensemble* |

You don't have to build this by hand.  `fp.add(df, name=...)` calls
`ForecastPerformance.normalize_dataframe` for you, which accepts **wide** frames
(datetime row index, `MultiIndex` columns) or **long** frames, normalises common
level-name aliases (`probability`/`prob`/`quantile` → `non_exceedance`,
`ensemble`/`member` → `ensemble_member`, `lead`/`lead_time` → `leadtime`, …), and
derives the missing one of `production_datetime` / `event_datetime` / `leadtime`
when the other two are present.

### Metrics are names *and* handles

Every metric is a `Metric` object — a callable that **stringifies to its own
name**.  This means you can pass it as a handle (`rmse`) or as a string
(`"rmse"`), and you can drop it straight into a results table without
`metric.__name__`:

```python
str(rmse) == "rmse"     # True
rmse == "rmse"          # True
rmse(forecast, obs)     # callable
```

Every metric is also a convenience **handle attribute** on the instance under
its common-usage name (acronyms uppercased like `fp.RMSE`, `fp.CRPS`,
`fp.fair_CRPS`; word-based metrics snake_case like `fp.reliability`), so you can
build a metrics list without importing anything:

```python
metrics = [fp.CRPS, fp.fair_CRPS, "reliability", "resolution"]
for metric in metrics:
    fp.probabilistic(metric, "prob_model", leadtime=lt)
```

---

## Quick start

```python
import pandas as pd
import numpy as np
from performance import ForecastPerformance, rmse, nse, crps

dates = pd.date_range("2020-01-01", periods=365, freq="D")
reference = pd.Series(
    np.sin(np.arange(365) * 2 * np.pi / 365), index=dates, name="Reference"
)
fp = ForecastPerformance(reference)
```

> **Quieting warnings.** Probabilistic CRPS warns when a forecast's CDF does not
> span `[0, 1]`. Pass `ForecastPerformance(reference, warn=False)` to silence
> these informative `UserWarning`s. (Spurious *numerical* warnings from the
> internal integrals are always suppressed.)

### Point (deterministic) forecast

```python
forecast = pd.DataFrame(
    reference.values + np.random.normal(0, 0.1, 365),
    index=dates,
    columns=pd.Index([pd.Timedelta("0D")], name="leadtime"),
)
fp.add(forecast, name="my_model")

# Three equivalent calling styles:
fp.deterministic(rmse, "my_model")       # metric handle
fp.deterministic("rmse", "my_model")     # metric name (or alias, e.g. "RMSE")
fp.deterministic.rmse("my_model")        # discoverable accessor (autocompletes)
```

### Quantile (probabilistic) forecast

```python
QUANTILE_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
quantile_df = pd.DataFrame(
    ...,  # shape (n_dates, n_quantiles)
    index=dates,
    columns=pd.MultiIndex.from_product(
        [[pd.Timedelta("0D")], QUANTILE_LEVELS],
        names=["leadtime", "non_exceedance"],
    ),
)
fp.add(quantile_df, name="prob_model")

lt = pd.Timedelta("0D")
fp.probabilistic(crps, "prob_model", leadtime=lt)      # handle
fp.probabilistic("crps", "prob_model", leadtime=lt)    # name
fp.probabilistic.crps("prob_model", leadtime=lt)       # accessor
```

### Ensemble forecast

```python
N_MEMBERS = 20
ensemble_df = pd.DataFrame(
    ...,  # shape (n_dates, N_MEMBERS)
    index=dates,
    columns=pd.MultiIndex.from_product(
        [[pd.Timedelta("0D")], range(N_MEMBERS)],
        names=["leadtime", "ensemble_member"],
    ),
)
fp.add(ensemble_df, name="ens_model")
fp.probabilistic.fair_crps("ens_model", leadtime=pd.Timedelta("0D"))
fp.probabilistic.brier_score("ens_model", leadtime=pd.Timedelta("0D"), threshold=0.5)
```

### Collecting results

`Results` accumulates rows and pivots them into a `DataFrame`.  Because metrics
stringify to their name, append the metric object directly:

```python
from performance import Results

results = Results("Model", "Metric", "Leadtime")
for name in fp.names():
    for metric in fp.deterministic.metrics:
        for lt in fp.simulations[name]["leadtimes"]:
            results.append(
                Model=name,
                Metric=metric,                              # no .__name__
                Leadtime=lt,
                Value=fp.deterministic(metric, name, leadtime=lt),
            )

table = results.to_pandas(index=["Metric", "Model"], columns=["Leadtime"])
```

### Baselines, corrections and housekeeping

```python
persistence = fp.get_persistence(leadtimes=pd.timedelta_range("0D", "10D", freq="1D"))
climatology = fp.get_climatology(rolling_window=30)
fp.add(climatology, name="climatology")

fp.adjust_mean("ens_model")     # shift ensemble mean to the reference mean
fp.adjust_scale("ens_model")    # scale ensemble mean to the reference mean

fp.clear_cache("ens_model")     # drop cached intermediates (force recompute)
fp.remove("climatology")        # delete a simulation entirely
```

### Visualisation

```python
import plotly.graph_objects as go
from performance import plotly_forecasting as gof

fp.qq_plot("prob_model")        # PIT / Q-Q calibration plot (matplotlib)

fig = go.Figure()
gof.plot_lt_probabilistic(fig, quantile_df_long, leadtimes=[pd.Timedelta("0D")])
gof.add_observed_trace(fig, reference)
gof.apply_default_layout(fig, yaxis_title="Q [m3/s]")
```

---

## Project structure

```
forecast_performance/
├── performance/
│   ├── __init__.py                 # Public re-exports (metrics, Metric, registries)
│   ├── forecast_performance.py     # ForecastPerformance main class
│   ├── results.py                  # Results accumulator class
│   ├── decorators.py               # storedResults caching decorator
│   ├── plotly_forecasting.py       # Plotly visualisation helpers
│   └── metrics/
│       ├── __init__.py             # metric exports + DETERMINISTIC/PROBABILISTIC registries
│       ├── base.py                 # Metric (callable that == its name)
│       ├── accessors.py            # fp.deterministic / fp.probabilistic accessors
│       ├── deterministic.py        # Pure deterministic metric functions
│       └── probabilistic.py        # Pure probabilistic metric functions
├── notebooks/                      # runnable usage examples (see "Notebooks" above)
│   ├── 00_visualize.ipynb          # Plotly forecast visualisation
│   ├── 01_benchmarks.ipynb         # persistence & climatology baselines
│   ├── 02_deterministic.ipynb      # deterministic metrics workflow
│   ├── 03_ensemble.ipynb           # ensemble metrics workflow
│   └── 04_probabilistic.ipynb      # probabilistic / quantile metrics workflow
├── tests/
│   ├── conftest.py                 # daily-parquet + synthetic fixtures
│   ├── test_forecast_performance.py
│   ├── test_deterministic.py
│   ├── test_probabilistic.py
│   ├── test_normalize.py
│   ├── test_results.py
│   ├── test_plotting.py
│   ├── test_missing_data.py
│   ├── test_datasets_daily/        # obs/det/ens/prob parquet datasets
│   └── test_datasets_hourly/
├── AGENTS.md                       # conventions for AI coding agents (canonical)
├── CLAUDE.md                       # → points to AGENTS.md
├── .github/copilot-instructions.md # → points to AGENTS.md
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Running the tests

```bash
pytest tests/ -v
```

---

## Building a distribution (wheel + sdist)

The package builds with the standard [PEP 517](https://peps.python.org/pep-0517/)
toolchain. Install the build extra and run the `build` frontend from the
repository root:

```bat
conda activate forecast_performance
pip install -e ".[build]"
python -m build
```

This produces both artifacts in `dist/`:

```
dist/
├── forecast_performance-0.5.0-py3-none-any.whl
└── forecast_performance-0.5.0.tar.gz
```

Install the wheel anywhere (no source checkout needed):

```bat
pip install dist/forecast_performance-0.5.0-py3-none-any.whl
```

Optionally check the metadata and (if you publish) upload with `twine`:

```bat
twine check dist/*
twine upload dist/*        # requires PyPI credentials
```

Notes:
- The version is set in [pyproject.toml](pyproject.toml) (`project.version`) and
  mirrored in `performance.__version__` — bump both together.
- Only the `performance` package is shipped; the `tests/` datasets and notebooks
  are excluded from the wheel.

---

## API reference

### `ForecastPerformance`

| Method | Description |
|---|---|
| `add(data, name, leadtime=0D, sort=True)` | Register a simulation (auto-normalised to canonical long format) |
| `normalize_dataframe(data, value_name="values")` | *(static)* Convert any reasonable wide/long frame to canonical long format |
| `deterministic(metric, name, leadtime=None)` | Apply a deterministic metric (handle **or** name) to the expected forecast |
| `deterministic.<metric>(name, leadtime=None)` | Per-metric accessor method (autocompletes) |
| `probabilistic(metric, name, leadtime, months=None, metric_kwargs=None)` | Apply a probabilistic metric (handle **or** name) |
| `probabilistic.<metric>(name, leadtime=None, ...)` | Per-metric accessor method with metric-specific kwargs |
| `get_expected(name, leadtime=None)` | Expected (mean) forecast as a long-format `DataFrame` |
| `get_expected_prediction(name)` | Expected forecast across all leadtimes |
| `get_persistence(leadtimes)` | Persistence baseline forecast |
| `get_climatology(...)` | Fourier-fitted climatological probabilistic baseline |
| `adjust_mean(name)` / `adjust_scale(name)` | Per-leadtime additive / multiplicative correction to the reference mean (ensemble or probabilistic) |
| `crop_event_dates(...)` / `crop_production_dates(...)` | Restrict the evaluation window |
| `names()` / `leadtimes()` | Registered simulations and their leadtimes |
| `remove(name)` | Delete a simulation and its cached results |
| `clear_cache(name=None)` | Clear cached intermediates (all simulations if `name` is `None`) |
| `qq_plot(name, leadtimes=None, plot=True, ax=None)` | PIT / Q-Q calibration plot |

### Standalone metric functions

All deterministic functions accept 1-D `array-like` arguments `(simulations, targets)`.

| Function | Range | Perfect |
|---|---|---|
| `rmse` | [0, ∞) | 0 |
| `mae` | [0, ∞) | 0 |
| `mse` | [0, ∞) | 0 |
| `bias` | (−∞, ∞) | 0 |
| `relative_bias` | (−∞, ∞) | 0 |
| `pearson` | [−1, 1] | 1 |
| `spearman` | [−1, 1] | 1 |
| `nse` | (−∞, 1] | 1 |
| `kge` | (−∞, 1] | 1 |
| `kge_prime` | (−∞, 1] | 1 |
| `count` | ℕ | — |

`snake_case` is the primary spelling; `PascalCase` aliases (`RMSE`, `NSE`,
`KGE`, `KGEprime`, …) are retained for backward compatibility.  The registries
`DETERMINISTIC_METRICS` / `PROBABILISTIC_METRICS` map every name and alias to its
`Metric`.

### `Results`

```python
from performance import Results, rmse

r = Results("Model", "Metric")
r.append(Model="A", Metric=rmse, Value=0.12)   # Metric stringifies to "rmse"
r.append(Model="B", Metric=rmse, Value=0.08)
df = r.to_pandas(index=["Model"], columns=["Metric"])
```

---

## License

See [LICENSE](LICENSE).
