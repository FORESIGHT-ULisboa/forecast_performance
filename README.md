<!-- The image and the notebook links below use absolute URLs on purpose: this
     README is the PyPI long_description, and PyPI resolves nothing
     repo-relative, so relative paths would render as broken links there. -->
<p align="center">
  <img src="https://raw.githubusercontent.com/FORESIGHT-ULisboa/forecast_performance/main/notebooks/foresight.png" alt="FORESIGHT" width="320">
</p>

# Forecast Performance

<p align="center">
  <a href="https://pypi.org/project/forecast-performance/"><img src="https://img.shields.io/pypi/v/forecast-performance.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/forecast-performance/"><img src="https://img.shields.io/pypi/pyversions/forecast-performance.svg" alt="Supported Python versions"></a>
  <a href="https://github.com/FORESIGHT-ULisboa/forecast_performance/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
</p>

**`forecast_performance`** is a Python library created by [FORESIGHT - Forecasting and Optimization for Resilient Environmental Systems through Investigation with Groundbreaking Hydrological Tools](https://foresight.tecnico.ulisboa.pt/) for evaluating the skill of
deterministic and probabilistic forecasting models.  It provides a single
unified interface — `ForecastPerformance` — that handles point, quantile, and
ensemble forecasts and exposes a rich set of metrics and visualisation tools.

> **New here? Start with the notebooks** in
> [`notebooks/`](https://github.com/FORESIGHT-ULisboa/forecast_performance/tree/main/notebooks),
> each a self-contained, runnable walkthrough:
> [`00_visualize`](https://github.com/FORESIGHT-ULisboa/forecast_performance/blob/main/notebooks/00_visualize.ipynb) (Plotly forecast plots) ·
> [`01_benchmarks`](https://github.com/FORESIGHT-ULisboa/forecast_performance/blob/main/notebooks/01_benchmarks.ipynb) (persistence & climatology
> baselines) · [`02_deterministic`](https://github.com/FORESIGHT-ULisboa/forecast_performance/blob/main/notebooks/02_deterministic.ipynb) ·
> [`03_ensemble`](https://github.com/FORESIGHT-ULisboa/forecast_performance/blob/main/notebooks/03_ensemble.ipynb) ·
> [`04_probabilistic`](https://github.com/FORESIGHT-ULisboa/forecast_performance/blob/main/notebooks/04_probabilistic.ipynb). Run them under the
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
| **Parquet I/O** | `PandasForecast` — round-trips `pd.DateOffset` leadtimes through parquet |
| **Utilities** | `Results` accumulator, `storedResults` caching decorator |

---

## Installation

Two paths: install the **released package from PyPI** (use the package), or
install **from source** (develop / run the notebooks and tests).

### A · Install from PyPI

```bat
pip install forecast-performance
```

That is all — the runtime dependencies (numpy, pandas, scipy, matplotlib, plotly,
pyarrow) are declared in the wheel, so pip resolves them automatically. Upgrading
later needs no URL: `pip install --upgrade forecast-performance` always picks up
the newest release.

> The **distribution** is `forecast-performance`; the **import** is
> `forecast_performance`:
>
> ```python
> from forecast_performance import ForecastPerformance
> ```

Every release is also attached to the
[GitHub releases page](https://github.com/FORESIGHT-ULisboa/forecast_performance/releases),
which is useful to pin an exact build without going through PyPI:

```bat
pip install https://github.com/FORESIGHT-ULisboa/forecast_performance/releases/download/v1.0.0/forecast_performance-1.0.0-py3-none-any.whl
```

### B · Install from source (development)

#### 1  Create a conda environment

```bat
conda create -n forecast_performance python=3.11
conda activate forecast_performance
```

#### 2  Install the package and all dependencies

From the repository root, run:

```bat
pip install -e ".[dev]"
```

#### 3  Register the Jupyter kernel

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

### Saving forecasts with `DateOffset` leadtimes (`PandasForecast`)

Seasonal / monthly forecasts often express *leadtime* as a `pd.DateOffset`
(`pd.DateOffset(months=1)`, `pd.DateOffset(years=1)`) instead of a `pd.Timedelta`,
because calendar months and years have a variable length.  Parquet cannot
serialize `DateOffset` objects stored in an index or in the columns, so a plain
`df.to_parquet(...)` raises.

`PandasForecast` is a drop-in `pd.DataFrame` subclass that fixes this: it encodes
each `DateOffset` in a `leadtime` level (whether in the index **or** the columns)
on write and restores it on read.  Use it exactly like a `DataFrame`:

```python
from forecast_performance import PandasForecast

PandasForecast(df).to_parquet("forecast.parquet")     # df has a DateOffset leadtime
back = PandasForecast.read_parquet("forecast.parquet")  # plain DataFrame by default
back.index.get_level_values("leadtime")[0]            # <DateOffset: months=1>
```

`read_parquet` returns a plain `pd.DataFrame` by default (so the subclass type
never leaks downstream); pass `to_pandas=False` to get a `PandasForecast` back.

It is fully backward compatible: a frame with a `Timedelta` (or no) leadtime is
written verbatim by the normal pandas writer, and a file written this way is still
readable by plain `pd.read_parquet` (the leadtime then holds the encoded strings).

---

## Quick start

```python
import pandas as pd
import numpy as np
from forecast_performance import ForecastPerformance, rmse, nse, crps

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
from forecast_performance import Results

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
from forecast_performance import plotly_forecasting as gof

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
├── forecast_performance/
│   ├── __init__.py                 # Public re-exports (metrics, Metric, registries)
│   ├── forecast_performance.py     # ForecastPerformance main class
│   ├── pandas_forecast.py          # PandasForecast (DateOffset-aware parquet I/O)
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
│   ├── test_datasets_daily/        # obs/det/ens/prob parquet datasets (used by conftest)
│   └── test_datasets_hourly/       # only used by its own aux notebook
├── AGENTS.md                       # conventions for AI coding agents (canonical)
├── CLAUDE.md                       # → points to AGENTS.md
├── .github/copilot-instructions.md # → points to AGENTS.md
├── .github/workflows/tests.yml     # pytest on push / PR (3.11, 3.12)
├── .github/workflows/release.yml   # build + publish to PyPI on tag push
├── .github/scripts/check_version.py # release guard: version declared consistently
├── pyproject.toml                  # packaging + pytest/coverage configuration
├── MANIFEST.in                     # keeps the 34 MB tests/ tree out of the sdist
├── LICENSE                         # MIT
└── README.md
```

---

## Running the tests

```bash
pytest tests/ -v
```

The suite also runs in CI on every push and pull request
([.github/workflows/tests.yml](.github/workflows/tests.yml)) against Python 3.11
and 3.12.

> The tests are **not** shipped in the PyPI sdist — their parquet fixtures are
> ~34 MB. Clone the repository, or download a tagged source archive, to run them.

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
├── forecast_performance-1.0.0-py3-none-any.whl
└── forecast_performance-1.0.0.tar.gz
```

Install the wheel anywhere (no source checkout needed):

```bat
pip install dist/forecast_performance-1.0.0-py3-none-any.whl
```

Check the metadata the way CI does:

```bat
twine check --strict dist/*
```

Notes:
- The version is set in [pyproject.toml](pyproject.toml) (`project.version`) and
  mirrored in `forecast_performance.__version__` — bump both together.
  [.github/scripts/check_version.py](.github/scripts/check_version.py) enforces
  that at release time.
- Only the `forecast_performance` package is shipped. [MANIFEST.in](MANIFEST.in)
  prunes `tests/`, whose parquet fixtures are ~34 MB.
- Building by hand is only needed to inspect an artifact locally, and **never
  `twine upload` by hand** — releases are produced by CI, which builds into
  `dist-release/` so the two never mix. See [Releasing](#releasing) below.

---

## Releasing

Releases are automated by
[.github/workflows/release.yml](.github/workflows/release.yml). To cut version
`X.Y.Z`:

1. Bump the version in **both** [pyproject.toml](pyproject.toml) and
   `forecast_performance.__version__`, refresh the pinned wheel URL under
   [Installation A](#a--install-from-pypi), then commit.
2. Rehearse if you want to: run the workflow from the **Actions** tab and leave the
   target as `testpypi`. It builds and uploads to
   [TestPyPI](https://test.pypi.org/project/forecast-performance/), no tag needed,
   and reruns are idempotent.
3. Tag and push:

   ```bat
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

The workflow then verifies the two version declarations agree with the tag, builds
the wheel and sdist, runs `twine check --strict`, publishes to
[PyPI](https://pypi.org/project/forecast-performance/), and attaches both artifacts
to the GitHub release.

> **PyPI uploads are immutable** — a version number can never be reused, even after
> deletion. Get the version right before pushing the tag, and never move or
> re-point a tag that has already been pushed. If a version has been consumed, bump
> and tag again.

### One-time setup (maintainers)

Publishing is authenticated with
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC):
GitHub mints a short-lived credential at publish time that PyPI accepts only for
this repository, this workflow file and this environment. **No API token is
created, stored or pasted anywhere.** Every value below is a public fact about the
repository — none of it is a secret, which is precisely the point.

**On PyPI** — once per index (repeat on [TestPyPI](https://test.pypi.org/) for the
rehearsal path):

1. *Your account → Publishing → Add a pending publisher* → GitHub, and fill in:

   | Field | Value |
   |---|---|
   | PyPI project name | `forecast-performance` |
   | Owner | `FORESIGHT-ULisboa` |
   | Repository | `forecast_performance` |
   | Workflow name | `release.yml` |
   | Environment | `pypi` (on TestPyPI: `testpypi`) |

   The environment name must match the `environment:` key of the corresponding job
   in [release.yml](.github/workflows/release.yml), or PyPI rejects the exchange. A
   *pending* publisher is what you register before the project exists on an index;
   it turns into an ordinary one on the first successful upload.

**On GitHub** — once:

2. *Settings → Environments* → create `pypi` (and `testpypi`).
3. Add a **required reviewer** to `pypi`. Every publish then pauses for an explicit
   approval, so a tag push alone cannot ship.

Things worth keeping that way:

- **Do not add a PyPI API token to repository secrets.** Trusted Publishing exists
  to remove that long-lived credential; a token in secrets is usable by any
  workflow in the repository and stays valid until someone revokes it. Nothing here
  needs one.
- The publisher trusts the workflow **by filename**, so anyone able to change
  [release.yml](.github/workflows/release.yml) or push a `v*` tag can trigger a
  publish. Protect `main` and restrict who may push tags; the environment reviewer
  from step 3 is the backstop.
- Only the two publish jobs request `id-token: write`; the build job is
  `contents: read` and the workflow default is `permissions: {}`. Keep that split
  when editing it — the job that produces the artifacts never holds the identity
  that can upload them.

---

## API reference

### `ForecastPerformance`

**Construction & data ingestion**

| Member | Description |
|---|---|
| `ForecastPerformance(reference, warn=True)` | Build an evaluator from an observation `Series`/`DataFrame`. `warn=False` silences informative `UserWarning`s (e.g. incomplete CDF boundaries). |
| `add(data, name, leadtime="0D", sort=True)` | Register a simulation; auto-normalised to canonical long format. `sort` enforces non-decreasing quantiles for probabilistic data. |
| `normalize_dataframe(data, value_name="values")` | *(static)* Convert any reasonable wide/long frame to canonical long format. |

**Scoring**

| Member | Description |
|---|---|
| `deterministic(metric, name, leadtime=None)` | Apply a deterministic metric (handle **or** name) to the expected forecast. |
| `deterministic.<metric>(name, leadtime=None)` | Per-metric accessor method (autocompletes), e.g. `fp.deterministic.rmse(...)`. |
| `deterministic.metrics` | List of the deterministic `Metric` objects (for iteration). |
| `probabilistic(metric, name, leadtime, months=None, metric_kwargs=None)` | Apply a probabilistic metric (handle **or** name). `months` filters by calendar month; `metric_kwargs` carries per-metric args (see below). |
| `probabilistic.<metric>(name, leadtime=None, ...)` | Per-metric accessor with metric-specific kwargs surfaced in the signature. |
| `probabilistic.metrics` | List of the probabilistic `Metric` objects. |

**Expected value & baselines**

| Member | Description |
|---|---|
| `get_expected(name, leadtime=None)` | Expected (mean) forecast as a long-format `DataFrame`. For probabilistic data the mean is integrated over the CDF. |
| `get_expected_prediction(name)` | Expected forecast across all registered leadtimes. |
| `get_persistence(leadtimes)` | Persistence baseline (observation at production time, carried forward). |
| `get_climatology(multiplicative=False, leadtimes=None, rolling_window=61, non_exceedance=None, coefficients=9, minimum=-inf, maximum=inf)` | Fourier-fitted seasonal cycle + empirical residual quantiles → probabilistic baseline. |

**Post-processing, cropping & management**

| Member | Description |
|---|---|
| `adjust_mean(name)` / `adjust_scale(name)` | Per-leadtime additive / multiplicative correction to the reference mean (ensemble **or** probabilistic; preserves member/quantile order). |
| `crop_event_dates(start=None, end=None)` / `crop_production_dates(start=None, end=None)` | Restrict the evaluation window by event / production date. |
| `names()` | List of registered simulation names. |
| `leadtimes()` | Boolean table of leadtimes × simulations. |
| `remove(name)` | Delete a simulation and its cached results. |
| `clear_cache(name=None)` | Clear cached intermediates (all simulations if `name` is `None`). |

**Visualisation**

| Member | Description |
|---|---|
| `qq_plot(name, leadtimes=None, plot=True, ax=None)` | PIT / Q-Q calibration plot; returns a `DataFrame` of `uniform`/`p_values`/`leadtime`. |

**Metric handle attributes** — every metric is also exposed on the instance/class
under its common-usage name, as a passable `Metric` handle (no import needed):
`fp.RMSE`, `fp.MAE`, `fp.MSE`, `fp.NSE`, `fp.KGE`, `fp.KGEprime`, `fp.Pearson`,
`fp.Spearman`, `fp.bias`, `fp.relative_bias`, `fp.count`, `fp.CRPS`,
`fp.fair_CRPS`, `fp.quantile_loss`, `fp.reliability`, `fp.resolution`,
`fp.resolution_relative`, `fp.brier_score`, `fp.fair_brier_score`,
`fp.fair_CRPS_skill_score`, `fp.fair_brier_skill_score`.

### Deterministic metrics

All accept 1-D `array-like` arguments `(simulations, targets)` and return a scalar.

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

### Probabilistic metrics

Applied through `fp.probabilistic(...)`. *Applies to* shows the simulation types
each supports (`simple` = point, `ens` = ensemble, `prob` = quantile).

| Metric | Applies to | Required args | Meaning |
|---|---|---|---|
| `crps` | simple, ens, prob | — | Continuous Ranked Probability Score (equals MAE for a point forecast). |
| `fair_crps` | ens (else = `crps`) | — | CRPS with the finite-ensemble-size bias removed. |
| `quantile_loss` | prob | — | Mean pinball loss averaged over quantile levels. |
| `reliability` | simple, ens, prob | — | PIT calibration index, range [−1, 1] (1 = perfectly calibrated). |
| `resolution` | ens, prob | — | Sharpness, `mean(1 / std)`. |
| `resolution_relative` | ens, prob | — | Relative sharpness, `mean(mean / std)`. |
| `brier_score` | simple, ens, prob | `threshold` | Brier score for the event "below `threshold`", range [0, 1]. Pass `return_p_values=True` to also get the exceedance probabilities. |
| `fair_brier_score` | ens | `threshold` | Brier score with the finite-ensemble correction. |
| `fair_crps_skill_score` | ens/prob | `reference` (+ `reference_leadtime`) | `1 − fairCRPS / fairCRPS_reference` vs a baseline simulation. |
| `fair_brier_skill_score` | ens/prob | `reference`, `threshold` (+ `reference_leadtime`) | `1 − fairBrier / fairBrier_reference`. |

Pass the required args either via `metric_kwargs=` on the generic call or as
keywords on the accessor method:

```python
fp.probabilistic("brier_score", "ens", leadtime=lt, metric_kwargs={"threshold": 100})
fp.probabilistic.brier_score("ens", leadtime=lt, threshold=100)

fp.probabilistic.fair_crps_skill_score("ens", leadtime=lt, reference="climatology")
```

Every probabilistic metric also accepts `months=[...]` to restrict the
evaluation to specific calendar months.

### `Metric`, registries and naming

- Each public metric is a `Metric` (subclass of `str`): callable, and equal to
  its own name (`str(rmse) == rmse == "rmse"`, `rmse.__name__ == "rmse"`).
- `snake_case` is the primary spelling; `PascalCase` aliases (`RMSE`, `NSE`,
  `KGE`, `KGEprime`, …) are retained for backward compatibility.
- `DETERMINISTIC_METRICS` / `PROBABILISTIC_METRICS` are dicts mapping every name
  **and** alias (case-insensitive) to its `Metric`; `DETERMINISTIC` /
  `PROBABILISTIC` are the ordered lists. All are importable from `forecast_performance`.

### `Results`

```python
from forecast_performance import Results, rmse

r = Results("Model", "Metric")
r.append(Model="A", Metric=rmse, Value=0.12)   # Metric stringifies to "rmse"
r.append(Model="B", Metric=rmse, Value=0.08)
df = r.to_pandas(index=["Model"], columns=["Metric"])
```

| Member | Description |
|---|---|
| `Results(*fields)` | Create an accumulator with the given field names (a `Value` field is added automatically). |
| `append(**values)` | Append one row (one keyword per field plus `Value`). |
| `to_pandas(index=None, columns=None)` | Pivot to a multi-indexed `DataFrame`; `index`/`columns` select which fields become row/column levels. |

### Visualisation helpers (`forecast_performance.plotly_forecasting`)

Plotly helpers that take a `go.Figure` and a canonical long-format frame:

| Function | Description |
|---|---|
| `plot_lt_deterministic(fig, df, leadtimes=None, **kw)` | Deterministic traces by leadtime. |
| `plot_pd_deterministic(fig, df, production_datetimes=None, **kw)` | Deterministic traces by production date. |
| `plot_lt_probabilistic(fig, df, leadtimes=None, bands=None, **kw)` | Shaded quantile bands by leadtime. |
| `plot_pd_probabilistic(fig, df, production_datetimes=None, **kw)` | Shaded quantile bands by production date. |
| `plot_pd_ensemble(fig, df, production_dates=None, ensembles=None, **kw)` | Ensemble member traces by production date. |
| `add_observed_trace(fig, obs, ...)` | Overlay the observation series. |
| `apply_default_layout(fig, yaxis_title="", ...)` | Apply the shared layout + range selector. |

### `PandasForecast`

A `pd.DataFrame` subclass whose parquet I/O preserves `pd.DateOffset` leadtimes
(which parquet cannot otherwise serialize). Each `DateOffset` in a `leadtime`
level — in the index or the columns — is encoded on write as a sentinel JSON
string of its `.kwds` and decoded back on read.

| Member | Description |
|---|---|
| `PandasForecast(data)` | Wrap a `DataFrame` (or anything `pd.DataFrame` accepts). |
| `to_parquet(path, *args, **kwargs)` | Like `DataFrame.to_parquet`, encoding any `DateOffset` leadtime first; delegates verbatim to the pandas writer when none is present. |
| `read_parquet(path, *args, to_pandas=True, **kwargs)` | *(classmethod)* Like `pd.read_parquet`, decoding any encoded leadtime. Returns a plain `pd.DataFrame` by default; pass `to_pandas=False` for a `PandasForecast`. |
| `to_pandas()` | Return a plain `pd.DataFrame` from a `PandasForecast` instance (decoded `DateOffset` leadtime preserved). Use it before handing data to downstream code that does exact-type checks. |

`PandasForecast` is a `pd.DataFrame` subclass, so it satisfies
`isinstance(x, pd.DataFrame)` and behaves like a frame everywhere. The subclass
type does propagate through operations (slicing, `groupby`, arithmetic, `concat`
all return `PandasForecast`); if downstream code relies on `type(x) is
pd.DataFrame`, unpickles without this package installed, or uses
`assert_frame_equal` with the frame as the *expected* argument, call
`to_pandas()` first.

Multi-keyword offsets (`pd.DateOffset(months=1, days=15)`) and mixed units across
leadtimes round-trip; offsets `.kwds` cannot capture (anchored `MonthEnd`, a bare
`pd.DateOffset(2)`) are left to the normal parquet behaviour.

### `storedResults`

Caching decorator used internally on `_p_values`; results are stored in
`fp.results[name][func][leadtime]` and bypassed when `threshold`/`months` is
supplied. Clear with `fp.clear_cache(...)`.

---

## License

Licensed under the **MIT License** — see
[LICENSE](https://github.com/FORESIGHT-ULisboa/forecast_performance/blob/main/LICENSE).
