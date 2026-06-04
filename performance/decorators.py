import inspect
import functools


def storedResults():
    """
    Caching decorator for :class:`ForecastPerformance` instance methods.

    Results are stored in ``obj.results[name][func_name][leadtime]``.
    The cache is **bypassed** whenever ``threshold`` or ``months`` is
    supplied — those calls always recompute and return immediately without
    writing to the cache.
    """

    def Inner(func):
        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs):
            arguments = inspect.signature(func).bind(obj, *args, **kwargs)
            arguments.apply_defaults()
            bound = arguments.arguments

            name = bound["name"]
            leadtime = bound.get("leadtime", None)
            threshold = bound.get("threshold", None)
            months = bound.get("months", None)

            fun_name = func.__name__
            if fun_name not in obj.results[name]:
                obj.results[name][fun_name] = {}

            # Non-cacheable: bypass when conditional parameters are given
            if threshold is not None or months is not None:
                return func(obj, *args, **kwargs)

            # Cacheable path
            if leadtime in obj.results[name][fun_name]:
                return obj.results[name][fun_name][leadtime]

            res = func(obj, *args, **kwargs)
            obj.results[name][fun_name][leadtime] = res
            return res

        return wrapper

    return Inner
