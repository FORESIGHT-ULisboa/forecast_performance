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

import pandas as pd

from .forecast_performance import _normalise_name

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
        """
        df = pd.read_parquet(path, *args, **kwargs)
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
