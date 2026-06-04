import pandas as pd


class Results:
    """
    Accumulates scalar results and organises them in a :class:`~pandas.DataFrame`.

    Parameters
    ----------
    *fields : str
        Names of the grouping fields (e.g. ``'Model'``, ``'Metric'``, ``'Leadtime'``).
        A ``'Value'`` field is added automatically.

    Example
    -------
    >>> results = Results('Model', 'Zone', 'Metric', 'Leadtime')
    >>> results.append(Model='TFT', Zone='A', Metric='RMSE', Leadtime=1, Value=0.5)
    >>> df = results.to_pandas(index=['Metric', 'Model'], columns=['Zone', 'Leadtime'])
    """

    def __init__(self, *fields):
        self.fields = list(fields)
        self.results = {f: [] for f in self.fields + ["Value"]}

    def append(self, **values):
        """Append one row of results."""
        for k, v in values.items():
            self.results[k].append(v)

    def to_pandas(self, index=None, columns=None):
        """
        Convert accumulated results to a multi-indexed :class:`~pandas.DataFrame`.

        Parameters
        ----------
        index : list of str, optional
            Subset of *fields* to keep as row index levels.
        columns : list of str, optional
            Fields to pivot into column levels.
        """
        if index is None:
            index = []
        if columns is None:
            columns = []

        df = pd.DataFrame(self.results).set_index(self.fields)
        df = df.unstack(columns)

        if len(index) > 0:
            tmp = df.index.to_frame()
            tmp = tmp.loc[:, index]
            df.index = pd.MultiIndex.from_frame(tmp)

        df.sort_index(inplace=True, axis=0)
        df.sort_index(inplace=True, axis=1)
        return df
