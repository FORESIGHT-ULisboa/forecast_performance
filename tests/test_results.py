"""Tests for the Results helper class."""

import pandas as pd
import pytest

from performance import Results


class TestResultsInit:
    def test_fields_stored(self):
        r = Results("Model", "Zone", "Metric")
        assert r.fields == ["Model", "Zone", "Metric"]

    def test_value_key_added(self):
        r = Results("Model")
        assert "Value" in r.results
        assert "Model" in r.results

    def test_initial_lists_empty(self):
        r = Results("A", "B")
        assert r.results["A"] == []
        assert r.results["Value"] == []


class TestResultsAppend:
    def test_single_row(self):
        r = Results("Model", "Metric")
        r.append(Model="TFT", Metric="RMSE", Value=1.5)
        assert r.results["Model"] == ["TFT"]
        assert r.results["Metric"] == ["RMSE"]
        assert r.results["Value"] == [1.5]

    def test_multiple_rows(self):
        r = Results("Model")
        r.append(Model="A", Value=1)
        r.append(Model="B", Value=2)
        assert len(r.results["Model"]) == 2
        assert r.results["Value"] == [1, 2]


class TestResultsToPandas:
    def _make(self):
        r = Results("Model", "Metric", "Leadtime")
        for model in ["A", "B"]:
            for metric in ["RMSE", "MAE"]:
                for lt in [1, 2]:
                    r.append(Model=model, Metric=metric, Leadtime=lt, Value=float(lt))
        return r

    def test_returns_dataframe(self):
        r = self._make()
        df = r.to_pandas(index=["Metric", "Model"], columns=["Leadtime"])
        assert isinstance(df, pd.DataFrame)

    def test_shape(self):
        r = self._make()
        df = r.to_pandas(index=["Metric", "Model"], columns=["Leadtime"])
        # 2 metrics × 2 models rows; 2 leadtimes columns (under 'Value')
        assert df.shape == (4, 2)

    def test_sorted_axes(self):
        r = self._make()
        df = r.to_pandas(index=["Metric", "Model"], columns=["Leadtime"])
        row_labels = [idx[0] for idx in df.index]
        assert row_labels == sorted(row_labels)

    def test_no_columns_specified(self):
        r = Results("Model")
        r.append(Model="X", Value=42)
        df = r.to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert "Value" in df.columns
