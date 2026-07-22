"""
PandasForecast - a ``pd.DataFrame`` subclass with DateOffset-aware parquet I/O.

Seasonal / monthly forecasts naturally express *leadtime* as a
:class:`pandas.DateOffset` (``pd.DateOffset(months=n)``, ``pd.DateOffset(years=n)``)
rather than a :class:`pandas.Timedelta`, because calendar months and years have a
variable length and must be *added to a production date* to land on the correct
day.  Parquet (via pyarrow / fastparquet) cannot serialize ``DateOffset`` objects
stored in an index or in the columns, so a plain ``df.to_parquet(...)`` raises.

:class:`PandasForecast` overrides :meth:`~PandasForecast.to_parquet` and adds a
:meth:`~PandasForecast.read_parquet` classmethod that make such leadtimes survive a
parquet round-trip.  Each ``DateOffset`` value in a ``leadtime`` level is encoded on
write as a sentinel-prefixed JSON string of its keyword arguments
(``pd.DateOffset(months=3)`` -> ``'DateOffset:{"months": 3}'``).  The integer
multiplier ``.n`` is part of the offset too -- ``pd.DateOffset(months=3) * 2`` means
*six* months (``n == 2``, ``kwds == {'months': 3}``) -- so it is captured under a
reserved ``"n"`` key whenever it is not ``1``
(``'DateOffset:{"months": 3, "n": 2}'``); the ``n == 1`` case keeps the bare-kwds
form for backward compatibility.  These are plain strings, which every parquet engine
serializes natively, so the normal ``DataFrame.to_parquet`` writer is used unchanged.
On read, values carrying the sentinel are decoded back to
``pd.DateOffset(n=n, **kwds)``.

Because every value is self-describing, no custom parquet key-value metadata is
needed, mixed units across leadtimes (some ``months``, some ``years``),
multi-keyword offsets (``pd.DateOffset(months=1, days=15)``) and scaled offsets
(``pd.DateOffset(months=3) * 2``, ``pd.DateOffset(2)``) round-trip for free, and the
encoding is engine-agnostic.  Offsets that are not exactly a ``pd.DateOffset``
(anchored offsets such as ``MonthEnd``) are left untouched and degrade to the normal
parquet behaviour.

:class:`PandasForecast` also provides :meth:`~PandasForecast.tz_conversion`, which
re-expresses a forecast frame from one timezone into another.  It anchors on the
``event_datetime`` level (deriving it, or ``leadtime``, from the others when
missing), preserves tz-awareness (naive in -> naive wall-clock out), leaves
``leadtime`` untouched as a duration, and collapses the duplicate rows a
daylight-saving fall-back produces.

Examples
--------
>>> from performance import PandasForecast
>>> pf = PandasForecast(df)            # df has a DateOffset 'leadtime' level
>>> pf.to_parquet("forecast.parquet")
>>> back = PandasForecast.read_parquet("forecast.parquet")
>>> back.index.get_level_values("leadtime")[0]
<DateOffset: months=3>
"""

import json
import warnings

import pandas as pd

from .forecast_performance import _LEVEL_ORDER, _normalise_name

# Prefix marking a leadtime value as an encoded ``pd.DateOffset``.
_SENTINEL = "DateOffset:"

# Keyword arguments accepted by ``pd.DateOffset`` (plural = relative, singular =
# absolute).  Used to validate a decoded payload so an arbitrary JSON-looking
# leadtime string is never mistaken for an encoded offset.
_VALID_OFFSET_KWARGS = frozenset(
    {
        "years", "months", "weeks", "days",
        "hours", "minutes", "seconds", "microseconds", "nanoseconds",
        "year", "month", "day", "weekday",
        "hour", "minute", "second", "microsecond", "nanosecond",
    }
)


# ---------------------------------------------------------------------------
# Offset <-> string encoding
# ---------------------------------------------------------------------------


def _coerce_scalar(obj):
    """``json.dumps`` fallback: turn a numpy scalar into its native Python scalar.

    Offsets are commonly built from integer columns (e.g.
    ``leadtime.dt.days.map(lambda x: pd.DateOffset(months=x))``), so ``.kwds``
    often holds ``numpy`` integers, which ``json`` cannot serialize on its own.
    """
    item = getattr(obj, "item", None)
    if callable(item):
        return item()
    raise TypeError("offset value %r is not JSON-serializable" % (obj,))


def _encode_offset(value):
    """Encode a *plain* ``pd.DateOffset`` as a sentinel JSON string.

    Both the keyword arguments (``.kwds``) **and** the integer multiplier
    (``.n``) are captured, because ``.kwds`` alone is not the full offset: a
    scaled offset such as ``pd.DateOffset(months=3) * 2`` has ``n == 2`` and
    ``kwds == {'months': 3}`` and means *six* months, while a bare
    ``pd.DateOffset(2)`` carries its whole meaning in ``.n`` with empty kwds.
    Reconstructing from ``kwds`` alone would silently drop the factor of ``n``.

    To keep the common ``n == 1`` case (e.g. ``pd.DateOffset(months=3)``) on-disk
    representation unchanged -- so existing files and the plain-``pd.read_parquet``
    contract still hold -- the multiplier is added under a reserved ``"n"`` key
    **only when** ``n != 1`` (``'DateOffset:{"months": 3, "n": 2}'``); for
    ``n == 1`` the payload is just the kwds (``'DateOffset:{"months": 3}'``).

    ``.kwds`` values may be numpy scalars (offsets built from integer columns);
    they are coerced to native Python scalars before serialization.

    Returns ``None`` when *value* is not exactly a ``pd.DateOffset`` (e.g. a
    ``Timedelta`` or an anchored ``MonthEnd``), or when it carries nothing to
    encode (``pd.DateOffset()`` with ``n == 1`` and empty kwds), or when ``.kwds``
    is not JSON-serializable. Such values are left untouched and degrade to the
    normal parquet behaviour.
    """
    if type(value) is pd.DateOffset:
        payload = dict(value.kwds)
        if value.n != 1:
            payload["n"] = value.n
        if not payload:
            return None
        try:
            encoded = json.dumps(payload, sort_keys=True, default=_coerce_scalar)
        except TypeError:
            return None
        return _SENTINEL + encoded
    return None


def _decode_offset(value):
    """Decode a sentinel JSON string back to a ``pd.DateOffset``.

    The reserved ``"n"`` key (if present) restores the integer multiplier; the
    remaining keys are the ``pd.DateOffset`` keyword arguments. Any value that is
    not a string carrying the sentinel, or whose payload is not a JSON object
    whose non-``"n"`` keys are all recognised by ``pd.DateOffset``, is returned
    unchanged.
    """
    if isinstance(value, str) and value.startswith(_SENTINEL):
        try:
            payload = json.loads(value[len(_SENTINEL):])
        except (ValueError, TypeError):
            return value
        if isinstance(payload, dict) and payload:
            n = payload.get("n", 1)
            kwds = {k: v for k, v in payload.items() if k != "n"}
            if (
                isinstance(n, int)
                and not isinstance(n, bool)
                and set(kwds) <= _VALID_OFFSET_KWARGS
            ):
                return pd.DateOffset(n=n, **kwds)
    return value


# ---------------------------------------------------------------------------
# Axis (index / columns) helpers
# ---------------------------------------------------------------------------


def _find_leadtime_level(axis):
    """Locate the ``leadtime`` level on an ``Index`` or ``MultiIndex``.

    Returns ``(name, position, is_multiindex)`` or ``None``.  Aliases (``lead``,
    ``lead_time``) are matched via :func:`~performance.forecast_performance._normalise_name`,
    consistent with the rest of the package.
    """
    if isinstance(axis, pd.MultiIndex):
        for position, name in enumerate(axis.names):
            if _normalise_name(name) == "leadtime":
                return name, position, True
        return None
    if _normalise_name(axis.name) == "leadtime":
        return axis.name, 0, False
    return None


def _replace_level_values(axis, position, is_multiindex, new_values):
    """Return a copy of *axis* with one level's values replaced.

    Level names (including ``None``) and the order of the other levels are
    preserved by rebuilding from the full per-row arrays.
    """
    if is_multiindex:
        arrays = [axis.get_level_values(i) for i in range(axis.nlevels)]
        arrays[position] = pd.Index(new_values, name=axis.names[position])
        return pd.MultiIndex.from_arrays(arrays, names=list(axis.names))
    return pd.Index(new_values, name=axis.name)


def _encode_axis(axis):
    """Return *axis* with the leadtime level encoded, or ``None`` if unchanged.

    The level is encoded only when **every** value is a cleanly representable
    ``pd.DateOffset``; otherwise the axis is left untouched (a Timedelta leadtime
    is a no-op pass-through; a non-representable offset degrades to the normal
    parquet writer, which raises).
    """
    found = _find_leadtime_level(axis)
    if found is None:
        return None
    _, position, is_multiindex = found
    values = axis.get_level_values(position)
    encoded = [_encode_offset(v) for v in values]
    if not encoded or any(e is None for e in encoded):
        return None
    return _replace_level_values(axis, position, is_multiindex, encoded)


def _decode_axis(axis):
    """Return *axis* with encoded leadtime values decoded, or ``None`` if unchanged."""
    found = _find_leadtime_level(axis)
    if found is None:
        return None
    _, position, is_multiindex = found
    values = axis.get_level_values(position)
    if not any(isinstance(v, str) and v.startswith(_SENTINEL) for v in values):
        return None
    decoded = [_decode_offset(v) for v in values]
    return _replace_level_values(axis, position, is_multiindex, decoded)


# ---------------------------------------------------------------------------
# Timedelta column-index read workaround (pyarrow <-> pandas 3)
# ---------------------------------------------------------------------------


def _is_timedelta_precision_error(err):
    """True if *err* is the pandas-3 precision-less-timedelta cast failure.

    A ``pd.Timedelta`` leadtime carried in the *columns* (as opposed to the
    index) hits this: the columns of a ``MultiIndex`` are stringified into the
    parquet field names, so the level's real dtype survives only in the
    ``column_indexes`` pandas metadata. On read, pyarrow's
    ``_reconstruct_columns_from_metadata`` restores it by first casting the raw
    (string) level to ``np.dtype("m8")`` -- a timedelta with *no precision* --
    before the correct cast to the recorded unit, and pandas 3 rejects that
    intermediate step with ``"Passing in 'timedelta' dtype with no precision
    is not allowed"``. Plain :func:`pandas.read_parquet` therefore raises before
    any of our decoding runs.
    """
    message = str(err)
    return "timedelta" in message and "no precision" in message


def _sanitise_timedelta_column_metadata(table):
    """Neutralise ``timedelta64`` column-index entries in *table*'s metadata.

    Rewrites only each affected column-index level's ``pandas_type`` (from
    ``"timedelta64"`` to ``"unicode"``) while **keeping** its ``numpy_type``
    (e.g. ``"timedelta64[us]"``). That makes pyarrow skip the precision-less
    intermediate ``astype`` and convert the level straight to the recorded unit
    via its final ``astype(numpy_type)``, yielding a proper ``TimedeltaIndex``
    with no fix-up left for us to do.

    Returns *table* unchanged when there is no such level, so this is a no-op for
    every other file.
    """
    metadata = table.schema.metadata or {}
    raw = metadata.get(b"pandas")
    if raw is None:
        return table
    pandas_metadata = json.loads(raw)
    changed = False
    for column_index in pandas_metadata.get("column_indexes", []):
        if column_index.get("pandas_type") == "timedelta64":
            column_index["pandas_type"] = "unicode"
            changed = True
    if not changed:
        return table
    new_metadata = dict(metadata)
    new_metadata[b"pandas"] = json.dumps(pandas_metadata).encode("utf-8")
    return table.replace_schema_metadata(new_metadata)


def _read_parquet_timedelta_safe(path, columns=None, filters=None, filesystem=None):
    """Read a parquet file whose columns carry a ``timedelta64`` level.

    Fallback for :meth:`PandasForecast.read_parquet` when plain
    :func:`pandas.read_parquet` trips over the pyarrow<->pandas-3 timedelta
    column-index bug (see :func:`_is_timedelta_precision_error`). Reads the arrow
    table directly, neutralises the offending metadata
    (:func:`_sanitise_timedelta_column_metadata`) and lets pyarrow do the
    conversion. Supports the common ``columns`` / ``filters`` / ``filesystem``
    read options.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(
        path, columns=columns, filters=filters, filesystem=filesystem
    )
    table = _sanitise_timedelta_column_metadata(table)
    return table.to_pandas()


# ---------------------------------------------------------------------------
# Timezone conversion
# ---------------------------------------------------------------------------

#: Aggregations accepted by ``tz_conversion`` for collapsing DST-duplicated rows.
_DUPLICATE_STRATEGIES = frozenset({"mean", "first", "last"})


def _convert_datetime_index(values, tz_from, tz_new):
    """Convert a datetime level from *tz_from* to *tz_new*, preserving awareness.

    A tz-naive input is interpreted as wall-clock time in *tz_from*: it is
    localized to *tz_from*, converted to *tz_new*, then stripped back to naive
    wall-clock time in *tz_new*, so a naive frame stays naive.  A tz-aware input
    is simply converted to *tz_new* (staying aware) and *tz_from* is ignored.

    Returns a :class:`pandas.DatetimeIndex`.  On the naive path a daylight-saving
    fall-back collapses two instants onto the same wall-clock timestamp (handled
    downstream by de-duplication) and a spring-forward leaves a gap.
    """
    index = pd.DatetimeIndex(values)
    was_naive = index.tz is None
    if was_naive:
        index = index.tz_localize(tz_from)
    index = index.tz_convert(tz_new)
    if was_naive:
        index = index.tz_localize(None)
    return index


def _screen_index(df):
    """Pre-screen and canonicalise *df*'s row index for tz conversion.

    Normalises index level names via :func:`_normalise_name`, promotes an unnamed
    datetime level to ``production_datetime`` (then ``event_datetime``) -- mirroring
    ``ForecastPerformance.normalize_dataframe`` -- and reports which of the
    ``production_datetime`` / ``event_datetime`` / ``leadtime`` levels are present
    in the row index.

    Warns (``UserWarning``) when a ``production`` / ``event`` datetime level is
    found only in the *columns* (tz conversion acts on the row index only), and
    raises ``ValueError`` when the row index carries no absolute datetime level to
    convert.

    Returns ``(df, present)``: *df* is the same frame with canonical index names
    and *present* is a ``dict`` mapping each of the three level names to a bool.
    """
    index = df.index
    is_multi = isinstance(index, pd.MultiIndex)
    names = (
        [_normalise_name(n) for n in index.names]
        if is_multi
        else [_normalise_name(index.name)]
    )

    # Promote an unnamed datetime level to production_datetime (then event_datetime).
    for position, name in enumerate(names):
        dtype = str(index.get_level_values(position).dtype)
        if name is None and dtype.startswith("datetime"):
            names[position] = (
                "production_datetime"
                if "production_datetime" not in names
                else "event_datetime"
            )

    if is_multi:
        df.index = index.set_names(names)
    else:
        df.index = index.rename(names[0])

    present = {
        level: level in names
        for level in ("production_datetime", "event_datetime", "leadtime")
    }

    column_names = (
        [_normalise_name(n) for n in df.columns.names]
        if isinstance(df.columns, pd.MultiIndex)
        else [_normalise_name(df.columns.name)]
    )
    for level in ("production_datetime", "event_datetime"):
        if not present[level] and level in column_names:
            warnings.warn(
                "tz_conversion operates on the row index; the '%s' level is in "
                "the columns and will not be converted." % level,
                UserWarning,
                stacklevel=3,
            )

    if not present["production_datetime"] and not present["event_datetime"]:
        raise ValueError(
            "tz_conversion needs a 'production_datetime' or 'event_datetime' level "
            "in the row index to convert; found index levels: %s." % names
        )

    return df, present


def _collapse_duplicates(df, strategy):
    """Collapse rows with duplicate full-index tuples using *strategy*.

    A daylight-saving fall-back maps two instants onto the same wall-clock
    timestamp, so after conversion the (event_datetime, leadtime, ...) key can
    repeat.  Grouping by every index level aggregates only those genuine
    duplicates; a frame with a unique index is returned untouched.  ``sort=False``
    keeps first-appearance order and avoids sorting unorderable ``DateOffset``
    leadtime levels.
    """
    if not df.index.duplicated().any():
        return df
    grouped = df.groupby(level=list(range(df.index.nlevels)), sort=False)
    return getattr(grouped, strategy)()


def _reorder_levels(df):
    """Restore the canonical ``_LEVEL_ORDER`` on *df*'s index without dropping any.

    Levels not in ``_LEVEL_ORDER`` (e.g. an unexpected extra level) are kept and
    appended after the canonical ones.  A single-level index, an already-ordered
    index, or any index that cannot be reordered unambiguously is returned as-is.
    """
    if not isinstance(df.index, pd.MultiIndex):
        return df
    names = list(df.index.names)
    new_order = [n for n in _LEVEL_ORDER if n in names] + [
        n for n in names if n not in _LEVEL_ORDER
    ]
    if new_order == names:
        return df
    try:
        return df.reorder_levels(new_order)
    except (KeyError, ValueError):
        return df


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class PandasForecast(pd.DataFrame):
    """A ``pd.DataFrame`` whose parquet I/O preserves ``DateOffset`` leadtimes.

    Use it exactly like a :class:`pandas.DataFrame`; only the parquet methods
    differ.  :meth:`read_parquet` is a classmethod (mirroring the module-level
    :func:`pandas.read_parquet`) and returns a :class:`PandasForecast`.

    A file written by :meth:`to_parquet` is still readable by plain
    :func:`pandas.read_parquet`; the leadtime column then holds the encoded
    strings (e.g. ``'DateOffset:{"months": 3}'``) instead of ``DateOffset`` objects.
    """

    # Keep the subclass type through pandas operations where possible.  Nothing
    # needs to survive as an instance attribute (the encoding lives on disk), so
    # the usual subclass attribute-loss caveat does not apply here.
    _metadata = []

    @property
    def _constructor(self):
        return PandasForecast

    def to_parquet(self, path=None, *args, **kwargs):
        """Write to parquet, encoding any ``DateOffset`` leadtime level first.

        When no encodable leadtime offset is present this delegates verbatim to
        :meth:`pandas.DataFrame.to_parquet` (full backward compatibility).
        Otherwise the encoding is applied to a copy -- ``self`` is never mutated.
        """
        new_index = _encode_axis(self.index)
        new_columns = _encode_axis(self.columns)

        if new_index is None and new_columns is None:
            return super().to_parquet(path, *args, **kwargs)

        encoded = pd.DataFrame(self)
        if new_index is not None:
            encoded.index = new_index
        if new_columns is not None:
            encoded.columns = new_columns
        return encoded.to_parquet(path, *args, **kwargs)

    @classmethod
    def read_parquet(cls, path, *args, to_pandas=True, **kwargs):
        """Read a parquet file, decoding any encoded ``DateOffset`` leadtime level.

        Behaves like :func:`pandas.read_parquet` (decoding any encoded leadtime
        back to ``pd.DateOffset``). By default (``to_pandas=True``) it returns a
        plain :class:`pandas.DataFrame`, so the subclass type never leaks into
        downstream code; pass ``to_pandas=False`` to get a :class:`PandasForecast`
        instead (e.g. to chain another :meth:`to_parquet`).

        A ``pd.Timedelta`` leadtime carried in the *columns* trips a
        pyarrow<->pandas-3 reconstruction bug in plain
        :func:`pandas.read_parquet` (see :func:`_is_timedelta_precision_error`);
        such files are transparently recovered via
        :func:`_read_parquet_timedelta_safe` (honouring the common ``columns`` /
        ``filters`` / ``filesystem`` options).
        """
        try:
            df = pd.read_parquet(path, *args, **kwargs)
        except ValueError as err:
            if not _is_timedelta_precision_error(err):
                raise
            df = _read_parquet_timedelta_safe(
                path,
                columns=kwargs.get("columns"),
                filters=kwargs.get("filters"),
                filesystem=kwargs.get("filesystem"),
            )
        new_index = _decode_axis(df.index)
        if new_index is not None:
            df.index = new_index
        new_columns = _decode_axis(df.columns)
        if new_columns is not None:
            df.columns = new_columns

        if to_pandas:
            return df
        else:
            return cls(df)

    def to_pandas(self):
        """Return a plain :class:`pandas.DataFrame` of the same data.

        ``PandasForecast`` is a ``pd.DataFrame`` subclass, so it passes
        ``isinstance(x, pd.DataFrame)`` checks and behaves like one everywhere.
        The subclass does, however, propagate through most operations (slicing,
        ``groupby``, arithmetic, ``concat`` all return ``PandasForecast``) and
        differs from a plain frame for ``type(x) is pd.DataFrame`` checks,
        :func:`pandas.testing.assert_frame_equal` with ``check_frame_type=True``
        (the default) when it is the *expected* argument, and unpickling (which
        needs this package importable). Call ``to_pandas()`` to hand a plain
        frame to downstream code where any of that matters. The leadtime level
        keeps its reconstructed ``pd.DateOffset`` values.
        """
        return pd.DataFrame(self)

    def tz_conversion(self, tz_from="UTC", tz_new="UTC", duplicates="mean"):
        """Re-express the forecast's datetime levels from *tz_from* to *tz_new*.

        The conversion **anchors on** ``event_datetime`` (the observation time):

        * If ``event_datetime`` is present it is the level that gets converted.
        * If it is absent it is derived first as
          ``production_datetime + leadtime``.
        * If neither ``event_datetime`` nor ``leadtime`` is present, there is
          nothing to anchor on and ``production_datetime`` is converted directly.
        * A missing ``leadtime`` is derived as
          ``event_datetime - production_datetime`` when needed to rebuild
          ``production_datetime``.

        ``production_datetime`` is deleted before the conversion (so a
        daylight-saving fall-back does not split the duplicate rows on it) and,
        when it was originally present, recreated at the end as
        ``event_datetime - leadtime``.  ``leadtime`` is a duration and is never
        timezone-converted; ``non_exceedance`` / ``ensemble_member`` and the value
        column(s) are preserved.

        Timezone-awareness is preserved: a tz-naive frame is interpreted as
        wall-clock time in *tz_from*, converted, and returned as naive wall-clock
        time in *tz_new*; a tz-aware frame is converted and stays aware (with
        *tz_from* ignored).  Both defaults (``"UTC"``) make this a no-op.

        On the naive path a daylight-saving transition can make some wall-clock
        times **vanish** (spring-forward -- left as a gap) or **occur twice**
        (fall-back -- collapsed).  *duplicates* selects how the duplicated rows are
        collapsed: ``"mean"`` (default), ``"first"`` or ``"last"``.

        Returns a new :class:`PandasForecast`; ``self`` is never mutated.
        """
        if duplicates not in _DUPLICATE_STRATEGIES:
            raise ValueError(
                "duplicates must be one of %s; got %r."
                % (sorted(_DUPLICATE_STRATEGIES), duplicates)
            )

        df = pd.DataFrame(self).copy()
        df, present = _screen_index(df)

        def _convert_level(frame, name):
            is_multi = isinstance(frame.index, pd.MultiIndex)
            position = list(frame.index.names).index(name) if is_multi else 0
            converted = _convert_datetime_index(
                frame.index.get_level_values(name), tz_from, tz_new
            )
            frame.index = _replace_level_values(
                frame.index, position, is_multi, converted
            )
            return frame

        anchor_on_production = (
            present["production_datetime"]
            and not present["event_datetime"]
            and not present["leadtime"]
        )

        if anchor_on_production:
            df = _convert_level(df, "production_datetime")
            df = _collapse_duplicates(df, duplicates)
        else:
            # Derive event_datetime (anchor) and, if needed, leadtime.
            if not present["event_datetime"]:
                event = df.index.get_level_values(
                    "production_datetime"
                ) + df.index.get_level_values("leadtime")
                df = df.assign(event_datetime=event).set_index(
                    "event_datetime", append=True
                )
            if not present["leadtime"] and present["production_datetime"]:
                lead = df.index.get_level_values(
                    "event_datetime"
                ) - df.index.get_level_values("production_datetime")
                df = df.assign(leadtime=lead).set_index("leadtime", append=True)

            had_production = present["production_datetime"]
            if had_production:
                df = df.droplevel("production_datetime")

            df = _convert_level(df, "event_datetime")
            df = _collapse_duplicates(df, duplicates)

            if had_production:
                production = df.index.get_level_values(
                    "event_datetime"
                ) - df.index.get_level_values("leadtime")
                df = df.assign(production_datetime=production).set_index(
                    "production_datetime", append=True
                )

        df = _reorder_levels(df)
        return self._constructor(df)
