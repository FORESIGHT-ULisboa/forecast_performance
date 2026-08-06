# AGENTS.md — conventions for AI coding agents

This file is the canonical instruction set for AI agents (Claude Code, GitHub
Copilot, etc.) working in this repository. `CLAUDE.md` and
`.github/copilot-instructions.md` are thin pointers to this file.

## What this project is

`forecast_performance` evaluates the skill of **deterministic**, **ensemble**
and **probabilistic** forecasts against a reference (observation) series. The
public surface is the `ForecastPerformance` class in
[forecast_performance/forecast_performance.py](forecast_performance/forecast_performance.py); pure
metric functions live in [forecast_performance/metrics/](forecast_performance/metrics/).

## Environment

- The project runs in a **conda environment named `forecast_performance`**.
  Use the same environment for the package, the tests, and the notebooks
  (register it as a Jupyter kernel named `forecast_performance`).
- Hint (a developer machine; **path will differ on other computers** — do not
  hard-code it, resolve the env yourself): the interpreter has been seen at
  `C:\Users\<your-user>\.conda\envs\forecast_performance\python.exe`. `conda`
  may not be on `PATH`; if so, either activate the env in a conda-aware shell
  or call that `python.exe` directly. Prefer `conda activate forecast_performance`
  when `conda` is available.
- Setup: `pip install -e ".[dev]"`. `pyarrow` is a **runtime** dependency (the
  `PandasForecast` parquet path imports it), so it arrives with a plain install;
  the `dev` extra adds `fastparquet` (alternative engine), Jupyter and the
  notebook widgets. `pip install -e ".[test]"` is the lighter set CI uses.
- Run tests with `pytest tests/ -v`.

## Build / test / release

Published on PyPI as **`forecast-performance`** (`pip install
forecast-performance`); the **import** package is `forecast_performance`. It was
renamed from `performance` in 1.0.0 — a plain `import performance` no longer works.

Releases are cut by
[.github/workflows/release.yml](.github/workflows/release.yml) on a `vX.Y.Z` tag
push: it verifies the version declarations agree with the tag, builds, runs
`twine check --strict`, publishes to PyPI, and attaches both artifacts to the
GitHub release. A manual run from the Actions tab targets TestPyPI by default, for
rehearsal. [.github/workflows/tests.yml](.github/workflows/tests.yml) runs the
suite on push and pull request against Python 3.11 and 3.12.

Releasing is therefore: bump the version in **both** places, commit,
`git tag vX.Y.Z`, `git push origin vX.Y.Z`. Nothing is uploaded by hand.
**PyPI versions are immutable** — a number can never be reused, even after
deletion — so a failing version guard is a hard stop, not a nuisance, and a pushed
tag is never moved or re-pointed.

The version lives in **two** places that must be bumped together:
`pyproject.toml` (`project.version`) and `forecast_performance.__version__`
([forecast_performance/__init__.py](forecast_performance/__init__.py)).
[.github/scripts/check_version.py](.github/scripts/check_version.py) enforces this
(plus tag agreement) before anything is published.

Build output is **not** committed: `dist/` (local builds) and `dist-release/` (what
CI builds and publishes) are both gitignored. CI deliberately builds into
`dist-release/` so a stale local `dist/` can never be graded or uploaded.

### One-time PyPI setup (Trusted Publishing)

Authentication is OIDC; **no API tokens exist in the repository or in secrets, and
none should be added** — see [README.md](README.md#one-time-setup-maintainers) for
the registration steps and the reasoning. Two constraints matter when touching the
workflow:

- The environment names (`pypi` / `testpypi`) are part of what PyPI verifies.
  Renaming a job's `environment:` key without re-registering the publisher breaks
  the OIDC exchange.
- `id-token: write` belongs only to the publish jobs. The build job stays
  `contents: read`, so the job that produces artifacts never holds the identity
  that can upload them.

## Canonical long format

Every forecast is stored as a **single-column** `DataFrame` with a `MultiIndex`
drawn from: `production_datetime`, `event_datetime`, `leadtime`,
`non_exceedance` (probabilistic), `ensemble_member` (ensemble).

- Add data with `fp.add(df, name=...)`. It calls
  `ForecastPerformance.normalize_dataframe`, which accepts wide **or** long
  frames, normalises level-name aliases (`probability`/`prob`/`quantile` →
  `non_exceedance`; `ensemble`/`member` → `ensemble_member`; `lead`/`lead_time`
  → `leadtime`; `production`/`event` variants), and derives the missing one of
  `production_datetime` / `event_datetime` / `leadtime`.
- The simulation **type** (`simple` / `ensemble` / `probabilistic`) is detected
  from the index levels and stored in `fp.simulations[name]`.
- Alignment with the reference is an inner join on `event_datetime` with
  `dropna` — so incomplete archives (missing production dates, leadtimes,
  members, quantile levels, or gappy observations) degrade gracefully. Keep it
  that way; there are dedicated tests in
  [tests/test_missing_data.py](tests/test_missing_data.py).

## Parquet I/O for `DateOffset` leadtimes

Seasonal/monthly forecasts express `leadtime` as a `pd.DateOffset`
(`pd.DateOffset(months=n)` / `years=n`) rather than a `pd.Timedelta`, because
calendar months/years have variable length and must be *added to a production
date*. Parquet (pyarrow/fastparquet) **cannot serialize `DateOffset`** objects in
an index or columns. `PandasForecast`
([forecast_performance/pandas_forecast.py](forecast_performance/pandas_forecast.py)) is a
`pd.DataFrame` subclass that fixes this:

- `PandasForecast(df).to_parquet(path)` finds the `leadtime` level (index **or**
  columns; aliases resolved via `_normalise_name`) and, if every value is a clean
  `pd.DateOffset`, encodes each as a sentinel-prefixed JSON string of its `.kwds`
  (`pd.DateOffset(months=3)` → `'DateOffset:{"months": 3}'`). The integer multiplier
  `.n` is part of the offset (`pd.DateOffset(months=3) * 2` means *six* months:
  `n == 2`, `kwds == {'months': 3}`), so it is captured under a reserved `"n"` key
  **only when `n != 1`** (`'DateOffset:{"months": 3, "n": 2}'`); the `n == 1` case
  keeps the bare-kwds form, so existing files and the plain-read contract are
  unchanged. Plain strings serialize natively, so the **normal pandas writer is used
  unchanged** — no custom parquet key-value metadata, no engine fork. `.kwds` values
  that are numpy scalars (offsets built from integer columns) are coerced to native
  Python scalars before JSON serialization (`_coerce_scalar`).
- `PandasForecast.read_parquet(path, *, to_pandas=True)` (a **classmethod**,
  mirroring the module-level `pd.read_parquet`) decodes those strings back to
  `pd.DateOffset` (via `pd.DateOffset(n=n, **kwds)`, restoring the `"n"` multiplier).
  It returns a **plain `pd.DataFrame` by default** (so the subclass doesn't leak
  downstream); pass `to_pandas=False` for a `PandasForecast`.
- Design choices to preserve: each value is self-describing, so multi-keyword
  offsets, mixed units across leadtimes, and scaled offsets (`pd.DateOffset(...) * 2`,
  bare `pd.DateOffset(2)`) round-trip for free. Values that are not exactly a
  `pd.DateOffset` (anchored offsets such as `MonthEnd`) are left untouched and degrade
  to the normal parquet behaviour. `to_parquet` never mutates `self`. A file written this way is still readable by plain
  `pd.read_parquet` (leadtime then holds the encoded strings) — keep that
  backward-compat contract; tests live in
  [tests/test_pandas_forecast.py](tests/test_pandas_forecast.py).
- `read_parquet` also transparently recovers files whose **columns** carry a
  `pd.Timedelta` leadtime level. A `MultiIndex`'s column names are stringified
  into the parquet field names, so that level's timedelta dtype survives only in
  pandas' `column_indexes` metadata; on read, pyarrow (with pandas 3)
  reconstructs it via a precision-less `astype(np.dtype("m8"))` that plain
  `pd.read_parquet` rejects (`"Passing in 'timedelta' dtype with no
  precision..."`). `read_parquet` catches that specific `ValueError` and retries
  through `_read_parquet_timedelta_safe`, which rewrites only the level's
  `pandas_type` (`timedelta64`→`unicode`) while keeping its `numpy_type`, so
  pyarrow restores the proper `TimedeltaIndex` itself. The recovery path honours
  the common `columns` / `filters` / `filesystem` options.
- It is a `pd.DataFrame` subclass (so `isinstance(x, pd.DataFrame)` holds and the
  type propagates through ops via `_constructor`). `to_pandas()` returns a plain
  `pd.DataFrame` for downstream code that does `type(x) is pd.DataFrame` checks,
  unpickles without this package, or uses `assert_frame_equal` with the frame as
  the expected arg. There is no extra instance state (`_metadata = []`), so the
  downcast is loss-free.

## Timezone conversion

`PandasForecast.tz_conversion(tz_from="UTC", tz_new="UTC", duplicates="mean")`
re-expresses a forecast frame from one timezone to another. It returns a new
`PandasForecast` and never mutates `self`; both defaults (`"UTC"`) make it a
no-op.

- It **anchors on `event_datetime`** (the observation time). When
  `event_datetime` is absent it is derived as `production_datetime + leadtime`;
  when only `production_datetime` is present (no `event_datetime`, no `leadtime`)
  that level is converted directly instead.
- `production_datetime` is **dropped before the conversion** (so a DST fall-back
  does not split the duplicate rows on it) and, when it was originally present,
  **recreated at the end** as `event_datetime - leadtime`. A missing `leadtime`
  needed for that step is derived as `event_datetime - production_datetime`.
- `leadtime` is a **duration** and is never timezone-converted;
  `non_exceedance` / `ensemble_member` and the value column(s) are preserved (the
  index is only ever re-ordered to `_LEVEL_ORDER`, never stripped of levels).
- **Awareness is preserved**: a tz-naive frame is read as wall-clock time in
  `tz_from`, converted, and returned as naive wall-clock time in `tz_new`
  (`tz_localize(tz_from).tz_convert(tz_new).tz_localize(None)`); a tz-aware frame
  is converted and stays aware (`tz_from` ignored).
- On the naive path a **DST transition** makes some wall-clock times vanish
  (spring-forward — left as a gap) or occur twice (fall-back). The duplicate rows
  are collapsed with `duplicates` (`"mean"` default / `"first"` / `"last"`).
- A pre-screen (`_screen_index`) normalises level-name aliases via
  `_normalise_name`, promotes an unnamed datetime level, warns if a
  `production`/`event` level lives only in the columns, and raises `ValueError`
  when the row index has no absolute datetime level. Tests live in
  [tests/test_pandas_forecast.py](tests/test_pandas_forecast.py) (`TestTzConversion`).

## The metric system

- Every public metric is a `Metric` (see
  [forecast_performance/metrics/base.py](forecast_performance/metrics/base.py)) — a `str`
  subclass that is also callable. It **equals and stringifies to its own
  name**: `str(rmse) == rmse == "rmse"`, `rmse.__name__ == "rmse"`, and
  `rmse(forecast, obs)` still works.
- Because of this, never write `metric.__name__` when building a `Results`
  table — append the `Metric` object directly and it stores as the name.
- `deterministic` and `probabilistic` are **callable accessors** (see
  [forecast_performance/metrics/accessors.py](forecast_performance/metrics/accessors.py)) set on
  each instance. All three styles are equivalent and supported — keep them
  symmetric when adding metrics:
  ```python
  fp.deterministic(rmse, name, leadtime=lt)     # handle
  fp.deterministic("rmse", name, leadtime=lt)   # name / alias (case-insensitive)
  fp.deterministic.rmse(name, leadtime=lt)      # per-metric accessor method
  ```
  The accessor's `__call__` forwards to the private `_apply_deterministic` /
  `_apply_probabilistic` methods on the class; resolution of handle-or-name is
  done by `_resolve_deterministic_metric` / `_resolve_probabilistic_metric`
  against the `DETERMINISTIC_METRICS` / `PROBABILISTIC_METRICS` registries in
  [forecast_performance/metrics/__init__.py](forecast_performance/metrics/__init__.py).
- Every metric — deterministic **and** probabilistic — is also exposed as a
  convenience **handle attribute** on the instance/class under its common-usage
  name (acronyms uppercased like `fp.RMSE`, `fp.NSE`, `fp.KGEprime`, `fp.CRPS`,
  `fp.fair_CRPS`; word-based metrics stay snake_case like `fp.reliability`,
  `fp.brier_score`). These are the `Metric` objects themselves, so they can be
  dropped into a metrics list and passed to `fp.deterministic(...)` /
  `fp.probabilistic(...)` without importing anything: `metrics = [fp.CRPS,
  fp.fair_CRPS, "reliability"]`. `str(fp.CRPS) == "crps"` — the visible attribute
  name is just a convenience; the metric's identity is its canonical name.

### Adding a new metric (do all of these)

1. Implement the pure function in `metrics/deterministic.py` or
   `metrics/probabilistic.py` (prefix the raw impl with `_` for deterministic,
   or wrap in place for probabilistic) and bind a `Metric(...)` with any
   aliases.
2. Add it to the `DETERMINISTIC` / `PROBABILISTIC` ordered list (this builds the
   registry automatically).
3. Export it from `metrics/__init__.py` and `forecast_performance/__init__.py`.
4. Add an explicit per-metric method on the matching accessor in
   `metrics/accessors.py` (so editors autocomplete it and its kwargs).
5. Add a convenience handle attribute on `ForecastPerformance` under its
   common-usage name (mirroring the `RMSE`/`CRPS` blocks at the top of the
   class).
6. For probabilistic metrics, add a dispatch branch in `_apply_probabilistic`.

## Caching, results, baselines

- `Results` ([forecast_performance/results.py](forecast_performance/results.py)) is a simple
  accumulator: `append(**fields)` then `to_pandas(index=[...], columns=[...])`.
- `storedResults` ([forecast_performance/decorators.py](forecast_performance/decorators.py))
  caches per-leadtime intermediates (PIT p-values) in
  `fp.results[name][func][leadtime]`. The cache is **bypassed** when `threshold`
  or `months` is supplied. Use `fp.clear_cache(name=None)` to drop cached
  intermediates and `fp.remove(name)` to delete a simulation entirely.
- Baselines: `fp.get_persistence(leadtimes)` and `fp.get_climatology(...)`
  return canonical long-format frames you feed back through `fp.add`.
- Corrections: `fp.adjust_mean(name)` / `fp.adjust_scale(name)` apply a
  per-leadtime additive / multiplicative shift so the forecast mean matches the
  reference mean. They operate on **ensemble and probabilistic** simulations
  (same constant per leadtime, preserving rank/quantile order); deterministic
  (`simple`) raises.
- Warnings: `ForecastPerformance(reference, warn=False)` silences informative
  `UserWarning`s (e.g. incomplete CDF boundaries in CRPS). Spurious numerical
  warnings inside the integrals are suppressed at the source (`np.errstate` /
  `warnings.catch_warnings` in `metrics/probabilistic.py`) — keep new numeric
  code equally quiet rather than letting benign divide/NaN warnings leak.

## Visualisation

- `fp.qq_plot(...)` is matplotlib-based.
- [forecast_performance/plotly_forecasting.py](forecast_performance/plotly_forecasting.py)
  provides Plotly helpers (`plot_lt_*`, `plot_pd_*`, `add_observed_trace`,
  `apply_default_layout`, colour helpers). Plotting tests run headless: set the
  matplotlib `Agg` backend and inspect `fig.data` / `fig.layout` rather than
  rendering. See [tests/test_plotting.py](tests/test_plotting.py).

## Tests

- Fixtures are in [tests/conftest.py](tests/conftest.py): **daily parquet**
  fixtures (`fp_det_daily` / `fp_ens_daily` / `fp_prob_daily`, plus raw
  `*_daily` frames) for realistic integration, and **synthetic** fixtures
  (`fp_simple` / `fp_ensemble` / `fp_probabilistic` / `fp_multi_leadtime`) for
  analytic-exact assertions (e.g. `CRPS == MAE` for a point forecast).
- When you touch the metric API, keep the "three calling styles agree" tests
  and the cache tests in
  [tests/test_forecast_performance.py](tests/test_forecast_performance.py)
  passing.
- **The sdist ships the library only** — [MANIFEST.in](MANIFEST.in) prunes
  `tests/`, which is ~34 MB of parquet. Before being tempted to
  `recursive-include tests`: only `tests/test_datasets_daily/` is read by the suite
  (`conftest.py`); `tests/test_datasets_hourly/` is 27 MB read by **no test at
  all**, only by its own aux notebook. A partially shipped suite is worse than
  none, because it looks runnable and is not.

## Style

- snake_case is the primary spelling; PascalCase metric aliases (`RMSE`, `NSE`,
  `KGE`, `KGEprime`, …) are kept for backward compatibility — preserve them.
- Line length 88; the repo uses `black` (`editor.formatOnSave` is on).
- Don't break the public re-exports in
  [forecast_performance/__init__.py](forecast_performance/__init__.py).
- **Licensed MIT.** The license is declared as a PEP 639 SPDX expression
  (`license = "MIT"` plus `license-files`), which is why `build-system.requires`
  pins `setuptools>=77` and why there is **no** `License :: OSI Approved :: ...`
  classifier — PyPI rejects metadata carrying both.
