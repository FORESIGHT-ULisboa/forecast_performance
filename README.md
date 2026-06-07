# Forecast Performance

**`forecast_performance`** is a Python library created by [FORESIGHT - Forecasting and Optimization for Resilient Environmental Systems through Investigation with Groundbreaking Hydrological Tools - ](https://foresight.tecnico.ulisboa.pt/) for evaluating the skill of
deterministic and probabilistic forecasting models.  It provides a single
unified interface — `ForecastPerformance` — that handles point, quantile, and
ensemble forecasts and exposes a rich set of metrics and visualisation tools.

---

## Features

| Category | Metrics / tools |
|---|---|
| **Deterministic** | RMSE, MAE, MSE, Bias, Relative bias, Pearson r, Spearman r, NSE, KGE, KGE' |
| **Probabilistic** | CRPS, Fair CRPS, Fair CRPSS, Brier score, Brier skill score, Quantile loss |
| **Reliability** | PIT / QQ plot, Reliability index, Resolution (sharpness) |
| **Post-processing** | `adjustMean`, `adjustScale` |
| **Visualisation** | `plot()`, `leadtimePlot()`, `QQPlot()` |
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


```bat
Open them directly in VS Code — select the **forecast_performance** kernel in the top-right kernel picker
```

or type

```bat
jupyter lab notebooks\
```

## Quick start

### Point (deterministic) forecast

```python
import pandas as pd
import numpy as np
from performance import ForecastPerformance, rmse, nse

dates = pd.date_range("2020-01-01", periods=365, freq="D")
reference = pd.Series(np.sin(np.arange(365) * 2 * np.pi / 365), index=dates, name="Reference")

# Forecast: single-column DataFrame with a Leadtime index level
forecast = pd.DataFrame(
    reference.values + np.random.normal(0, 0.1, 365),
    index=dates,
    columns=pd.Index([pd.Timedelta("0D")], name="Leadtime"),
)

fp = ForecastPerformance(reference)
fp.add_by_production_date(forecast, name="my_model")

print("RMSE:", fp.deterministic(rmse, "my_model"))
print("NSE: ", fp.deterministic(nse,  "my_model"))
```

### Quantile forecast

```python
from performance import ForecastPerformance

QUANTILE_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]

quantile_df = pd.DataFrame(
    ...,  # shape (n_dates, n_quantiles)
    index=dates,
    columns=pd.MultiIndex.from_product(
        [[pd.Timedelta("0D")], QUANTILE_LEVELS],
        names=["Leadtime", "Probability"],
    ),
)

fp.add_by_production_date(quantile_df, name="prob_model")
print("CRPS:", fp.CRPS("prob_model", leadtime=pd.Timedelta("0D")))
```

### Ensemble forecast

```python
N_MEMBERS = 20
ensemble_df = pd.DataFrame(
    ...,  # shape (n_dates, N_MEMBERS)
    index=dates,
    columns=pd.MultiIndex.from_product(
        [[pd.Timedelta("0D")], range(N_MEMBERS)],
        names=["Leadtime", "Ensemble member"],
    ),
)

fp.add_by_production_date(ensemble_df, name="ens_model")
print("Fair CRPS:", fp.fairCRPS("ens_model", reference="ens_model",
                                leadtime=pd.Timedelta("0D")))
```

---

## Project structure

```
forecast_performance/
├── performance/
│   ├── __init__.py                 # Public re-exports
│   ├── decorators.py               # storedResults caching decorator
│   ├── results.py                  # Results accumulator class
│   ├── forecast_performance.py     # ForecastPerformance main class
│   └── metrics/
│       ├── __init__.py
│       ├── deterministic.py        # Pure deterministic metric functions
│       └── probabilistic.py        # Pure probabilistic metric functions
├── tests/
│   ├── conftest.py
│   ├── test_results.py
│   ├── test_deterministic.py
│   ├── test_probabilistic.py
│   └── test_forecast_performance.py
├── notebooks/
│   ├── 01_deterministic.ipynb
│   ├── 02_probabilistic.ipynb
│   └── 03_ensemble.ipynb
├── pyproject.toml
└── README.md
```

---

## Running the tests

```bash
pytest tests/ -v
```

---

## API reference

### `ForecastPerformance`

| Method | Description |
|---|---|
| `add_by_production_date(data, name)` | Register a simulation keyed by production date |
| `add_by_event_date(data, name)` | Register a simulation keyed by event date |
| `deterministic(metric, name, leadtime)` | Apply any metric function to the expected forecast |
| `CRPS(name, leadtime, ...)` | Continuous Ranked Probability Score |
| `fairCRPS(name, reference, leadtime)` | Fair CRPSS relative to a reference simulation |
| `BrierS(name, threshold, leadtime)` | Brier score for threshold exceedance |
| `fairBrierS(name, threshold, leadtime)` | Fair Brier skill score |
| `reliability(name, leadtimes)` | Reliability index (calibration) |
| `resolution(name, leadtimes)` | Resolution / sharpness |
| `get_expected_value(name, leadtime)` | Ensemble/quantile mean as a Series |
| `adjustMean(name)` | Add bias-corrected copy |
| `adjustScale(name)` | Add spread-corrected copy |
| `plot(name, ...)` | Forecast envelope plot |
| `leadtimePlot(name, metric, ...)` | Skill vs leadtime line plot |
| `QQPlot(name, leadtimes, ...)` | PIT / QQ calibration plot |

### Standalone metric functions

All functions accept 1-D `array-like` arguments `(simulations, targets)`.

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

### `Results`

```python
from performance import Results

r = Results("Model", "Metric")
r.append(Model="A", Metric="RMSE", Value=0.12)
r.append(Model="B", Metric="RMSE", Value=0.08)
df = r.to_pandas(index=["Model"], columns=["Metric"])
```

---

## License

See [LICENSE](LICENSE).
