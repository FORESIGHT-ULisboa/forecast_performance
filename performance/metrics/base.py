"""
The :class:`Metric` wrapper.

A :class:`Metric` is a thin, callable wrapper around a metric implementation
that **is its own name**.  It subclasses :class:`str`, so

* ``str(rmse) == rmse == "rmse"`` — it stringifies to its name,
* ``rmse.__name__ == "rmse"`` — the ``__name__`` attribute still works,
* it hashes, sorts and pivots inside a :class:`~pandas.DataFrame` exactly like
  the plain name string (handy for :class:`~performance.Results`),
* ``rmse(data, reference)`` still calls the underlying implementation.

This lets callers pass either the handle (``rmse``) or the name (``"rmse"``)
interchangeably, and drop the metric straight into a results table without
reaching for ``metric.__name__``.
"""


class Metric(str):
    """A callable metric that equals its own name.

    Parameters
    ----------
    name : str
        Canonical metric name (what the object stringifies to).
    func : callable
        Underlying implementation invoked on ``__call__``.
    kind : str
        ``"deterministic"`` or ``"probabilistic"``.
    aliases : iterable of str, optional
        Alternative names recognised when resolving a metric from a string.
    """

    def __new__(cls, name, func, kind="", aliases=()):
        obj = super().__new__(cls, name)
        obj._func = func
        obj.__name__ = name
        obj.__doc__ = getattr(func, "__doc__", None)
        obj.kind = kind
        obj.aliases = tuple(aliases)
        return obj

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def __repr__(self):
        return self.__name__

    # ``str`` is immutable; reconstruct from the name on (deep)copy/pickle by
    # falling back to a plain string, which compares equal to this Metric.
    def __reduce__(self):
        return (str, (str(self),))
