See [AGENTS.md](../AGENTS.md) for this project's conventions, architecture, the
metric/accessor system, the conda environment, and testing guidance. It is the
single source of truth for AI coding agents working in this repository.

Key reminders:
- Metrics are `Metric` objects that stringify to their name — pass `rmse` or
  `"rmse"` interchangeably and never write `metric.__name__` in `Results`.
- `fp.deterministic` / `fp.probabilistic` are callable accessors; the handle,
  name, and `fp.deterministic.rmse(...)` styles must stay equivalent.
- Use the `forecast_performance` conda environment for code, tests, and
  notebooks. Run tests with `pytest tests/ -v`.
