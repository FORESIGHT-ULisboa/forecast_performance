"""Tests for PandasForecast - DateOffset-aware parquet round-trips."""

import numpy as np
import pandas as pd
import pytest

from performance import PandasForecast
from performance.pandas_forecast import _SENTINEL


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

DATES = pd.date_range("2020-01-01", periods=6, freq="D")


def _long_frame(leadtimes):
    """Single-column long frame with a (production_datetime, leadtime) index.

    *leadtimes* is a list of leadtime values; the frame has one row per
    (date, leadtime) pair.  Built in one shot (no concat) so incomparable
    ``DateOffset`` levels don't trigger a spurious sort warning.
    """
    n = len(DATES)
    prod = np.tile(np.asarray(DATES), len(leadtimes))
    lts = [lt for lt in leadtimes for _ in range(n)]
    mi = pd.MultiIndex.from_arrays(
        [prod, lts], names=["production_datetime", "leadtime"]
    )
    values = np.tile(np.arange(n, dtype=float), len(leadtimes))
    return pd.DataFrame({"values": values}, index=mi)


def _leadtime_level(df):
    return df.index.get_level_values("leadtime")


# ---------------------------------------------------------------------------
# Round-trips that restore DateOffset
# ---------------------------------------------------------------------------


class TestRoundTripOffsets:
    def test_months(self, tmp_path):
        df = _long_frame([pd.DateOffset(months=1), pd.DateOffset(months=3)])
        path = tmp_path / "months.parquet"
        PandasForecast(df).to_parquet(path)
        back = PandasForecast.read_parquet(path)

        restored = sorted({o.kwds["months"] for o in _leadtime_level(back)})
        assert restored == [1, 3]
        assert all(isinstance(o, pd.DateOffset) for o in _leadtime_level(back))

    def test_years(self, tmp_path):
        df = _long_frame([pd.DateOffset(years=1), pd.DateOffset(years=2)])
        path = tmp_path / "years.parquet"
        PandasForecast(df).to_parquet(path)
        back = PandasForecast.read_parquet(path)

        restored = sorted({o.kwds["years"] for o in _leadtime_level(back)})
        assert restored == [1, 2]

    def test_mixed_units_and_multi_keyword(self, tmp_path):
        leadtimes = [
            pd.DateOffset(months=1),
            pd.DateOffset(years=1),
            pd.DateOffset(months=1, days=15),
        ]
        df = _long_frame(leadtimes)
        path = tmp_path / "mixed.parquet"
        PandasForecast(df).to_parquet(path)
        back = PandasForecast.read_parquet(path)

        restored = {tuple(sorted(o.kwds.items())) for o in _leadtime_level(back)}
        expected = {tuple(sorted(o.kwds.items())) for o in leadtimes}
        assert restored == expected

    def test_values_preserved(self, tmp_path):
        df = _long_frame([pd.DateOffset(months=2)])
        path = tmp_path / "vals.parquet"
        PandasForecast(df).to_parquet(path)
        back = PandasForecast.read_parquet(path)
        np.testing.assert_array_equal(back["values"].values, df["values"].values)

    def test_numpy_int_offsets(self, tmp_path):
        # Offsets built from an integer column carry numpy ints in .kwds, which
        # json cannot serialize natively -- they must be coerced transparently.
        leadtimes = [pd.DateOffset(months=np.int64(k)) for k in (1, 6)]
        df = _long_frame(leadtimes)
        path = tmp_path / "npint.parquet"
        PandasForecast(df).to_parquet(path)
        back = PandasForecast.read_parquet(path)
        restored = sorted({o.kwds["months"] for o in _leadtime_level(back)})
        assert restored == [1, 6]

    def test_scaled_offset_multiplier_n(self, tmp_path):
        # A scaled offset carries its factor in .n: pd.DateOffset(months=3) * 2
        # has n == 2, kwds == {'months': 3} and means SIX months.  Encoding
        # .kwds alone would silently drop the factor of n (reconstruct 3 months).
        three, six = pd.DateOffset(months=3), pd.DateOffset(months=3) * 2
        assert six.n == 2 and six.kwds == {"months": 3}
        df = _long_frame([three, six])
        path = tmp_path / "scaled.parquet"
        PandasForecast(df).to_parquet(path)
        back = PandasForecast.read_parquet(path)

        anchor = pd.Timestamp("2020-01-01")
        landed = sorted({(anchor + o) for o in _leadtime_level(back)})
        # 3 months -> April, 6 months -> July (NOT two April landings).
        assert landed == [pd.Timestamp("2020-04-01"), pd.Timestamp("2020-07-01")]

    def test_bare_offset_count_in_n(self, tmp_path):
        # pd.DateOffset(2) has empty kwds and n == 2; its whole meaning is in .n.
        bare = pd.DateOffset(2)
        assert bare.n == 2 and not bare.kwds
        df = _long_frame([bare])
        path = tmp_path / "bare.parquet"
        PandasForecast(df).to_parquet(path)
        back = PandasForecast.read_parquet(path)
        restored = _leadtime_level(back)[0]
        assert restored.n == 2 and not restored.kwds
        assert pd.Timestamp("2020-01-01") + restored == pd.Timestamp("2020-01-03")

    def test_n_one_disk_format_unchanged(self, tmp_path):
        # The n == 1 common case must keep the bare-kwds on-disk form (no "n"
        # key), preserving backward compatibility and the plain-read contract.
        df = _long_frame([pd.DateOffset(months=3)])
        path = tmp_path / "n1.parquet"
        PandasForecast(df).to_parquet(path)
        plain = pd.read_parquet(path)
        assert plain.index.get_level_values("leadtime")[0] == _SENTINEL + '{"months": 3}'

    def test_alias_lead_time(self, tmp_path):
        df = _long_frame([pd.DateOffset(months=4)])
        df.index = df.index.set_names(["production_datetime", "lead_time"])
        path = tmp_path / "alias.parquet"
        PandasForecast(df).to_parquet(path)
        back = PandasForecast.read_parquet(path)
        offsets = back.index.get_level_values("lead_time")
        assert offsets[0].kwds == {"months": 4}


# ---------------------------------------------------------------------------
# Wide format (leadtime as a column level)
# ---------------------------------------------------------------------------


class TestWideColumns:
    def test_leadtime_in_columns(self, tmp_path):
        # Realistic 2-level wide shape: leadtime + ensemble_member.
        leadtimes = [pd.DateOffset(months=1), pd.DateOffset(months=2)]
        cols = pd.MultiIndex.from_product(
            [leadtimes, range(3)], names=["leadtime", "ensemble_member"]
        )
        df = pd.DataFrame(
            np.random.default_rng(0).normal(size=(len(DATES), len(cols))),
            index=DATES,
            columns=cols,
        )
        df.index.name = "production_datetime"
        path = tmp_path / "wide.parquet"
        PandasForecast(df).to_parquet(path)
        back = PandasForecast.read_parquet(path)

        restored = sorted({o.kwds["months"] for o in back.columns.get_level_values("leadtime")})
        assert restored == [1, 2]


# ---------------------------------------------------------------------------
# Pass-through / backward compatibility
# ---------------------------------------------------------------------------


class TestPassThrough:
    def test_timedelta_leadtime_unchanged(self, tmp_path):
        df = _long_frame([pd.Timedelta("1D"), pd.Timedelta("2D")])
        path = tmp_path / "td.parquet"
        PandasForecast(df).to_parquet(path)

        # No sentinel encoding written; plain pandas reads it back identically.
        plain = pd.read_parquet(path)
        assert not any(
            isinstance(v, str) and v.startswith(_SENTINEL)
            for v in plain.index.get_level_values("leadtime")
        )
        back = PandasForecast.read_parquet(path)
        assert list(_leadtime_level(back).unique()) == [
            pd.Timedelta("1D"),
            pd.Timedelta("2D"),
        ]

    def test_timedelta_leadtime_in_columns(self, tmp_path):
        # A Timedelta leadtime carried in a *column* level round-trips. The
        # columns of a MultiIndex are stringified into the parquet field names,
        # so the level's timedelta dtype survives only in pandas' column_indexes
        # metadata; on read, pyarrow (with pandas 3) reconstructs it via a
        # precision-less astype that plain pd.read_parquet chokes on. read_parquet
        # must transparently recover such files.
        leadtimes = [pd.Timedelta("1D"), pd.Timedelta("2D"), pd.Timedelta("3D")]
        cols = pd.MultiIndex.from_product(
            [leadtimes, range(2)], names=["leadtime", "ensemble_member"]
        )
        df = pd.DataFrame(
            np.random.default_rng(1).normal(size=(len(DATES), len(cols))),
            index=DATES,
            columns=cols,
        )
        df.index.name = "production_datetime"
        path = tmp_path / "td_columns.parquet"
        PandasForecast(df).to_parquet(path)

        back = PandasForecast.read_parquet(path)
        leadtime_level = back.columns.get_level_values("leadtime")
        assert pd.api.types.is_timedelta64_dtype(leadtime_level)
        assert list(leadtime_level.unique()) == leadtimes
        np.testing.assert_array_equal(back.to_numpy(), df.to_numpy())

    def test_no_leadtime_level(self, tmp_path):
        df = pd.DataFrame({"values": [1.0, 2.0, 3.0]}, index=DATES[:3])
        df.index.name = "production_datetime"
        path = tmp_path / "noleadtime.parquet"
        PandasForecast(df).to_parquet(path)
        back = PandasForecast.read_parquet(path)
        np.testing.assert_array_equal(back["values"].values, df["values"].values)

    def test_existing_dataset_roundtrips(self, tmp_path, ens_daily):
        """A real Timedelta-indexed dataset survives PandasForecast I/O."""
        path = tmp_path / "ens.parquet"
        PandasForecast(ens_daily).to_parquet(path)
        back = PandasForecast.read_parquet(path)
        pd.testing.assert_frame_equal(pd.DataFrame(back), pd.DataFrame(ens_daily))


# ---------------------------------------------------------------------------
# On-disk representation and types
# ---------------------------------------------------------------------------


class TestContract:
    def test_plain_read_sees_encoded_strings(self, tmp_path):
        df = _long_frame([pd.DateOffset(months=3)])
        path = tmp_path / "encoded.parquet"
        PandasForecast(df).to_parquet(path)
        plain = pd.read_parquet(path)
        values = plain.index.get_level_values("leadtime")
        assert all(isinstance(v, str) and v.startswith(_SENTINEL) for v in values)
        assert values[0] == _SENTINEL + '{"months": 3}'

    def test_read_returns_plain_frame_by_default(self, tmp_path):
        df = _long_frame([pd.DateOffset(months=1)])
        path = tmp_path / "type.parquet"
        PandasForecast(df).to_parquet(path)
        # Default to_pandas=True -> a plain DataFrame, not the subclass.
        result = PandasForecast.read_parquet(path)
        assert type(result) is pd.DataFrame

    def test_read_to_pandas_false_returns_subclass(self, tmp_path):
        df = _long_frame([pd.DateOffset(months=1)])
        path = tmp_path / "type_sub.parquet"
        PandasForecast(df).to_parquet(path)
        result = PandasForecast.read_parquet(path, to_pandas=False)
        assert isinstance(result, PandasForecast)

    def test_to_pandas_returns_plain_frame(self, tmp_path):
        df = _long_frame([pd.DateOffset(months=2)])
        path = tmp_path / "topandas.parquet"
        PandasForecast(df).to_parquet(path)
        plain = PandasForecast.read_parquet(path, to_pandas=False).to_pandas()
        assert type(plain) is pd.DataFrame
        # DateOffset leadtime survives the downcast.
        assert _leadtime_level(plain)[0].kwds == {"months": 2}

    def test_to_parquet_does_not_mutate_self(self, tmp_path):
        df = _long_frame([pd.DateOffset(months=1)])
        pf = PandasForecast(df)
        PandasForecast.to_parquet(pf, tmp_path / "nomutate.parquet")
        # Original leadtime level still holds DateOffset objects.
        assert isinstance(pf.index.get_level_values("leadtime")[0], pd.DateOffset)


# ---------------------------------------------------------------------------
# Degrade behaviour for non-representable offsets
# ---------------------------------------------------------------------------


class TestDegrade:
    def test_anchored_offset_is_not_encoded(self, tmp_path):
        df = _long_frame([pd.offsets.MonthEnd(1)])
        path = tmp_path / "anchored.parquet"
        # MonthEnd is not encodable -> falls through to the normal writer, which
        # cannot serialize the offset and raises.
        with pytest.raises(Exception):
            PandasForecast(df).to_parquet(path)
