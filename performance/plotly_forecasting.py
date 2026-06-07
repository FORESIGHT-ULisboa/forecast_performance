import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re

default_color_scale = px.colors.sequential.Plasma

update_xaxes = dict(rangeslider_visible=True,
    rangeselector=dict(
        buttons=[
            dict(count=7, label="7d", step="day", stepmode="backward"),
            dict(count=14, label="14d", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all", label="all"),
        ]
))

update_layout = dict(template="seaborn",
    width=1000,
    height=550,
    hovermode="x unified",
    margin=dict(l=60, r=30, t=40, b=60),
)

def color_sampler(n, colors=None, color_loops=1, **kwargs):
    if colors is None:
        colors = default_color_scale

    if n <= 0:
        return

    n_base = max(1, round(n / color_loops))

    positions = [0.5] if n_base == 1 else [i / (n_base - 1) for i in range(n_base)]
    base_colors = px.colors.sample_colorscale(colors, positions)

    for i in range(n):
        yield base_colors[i % n_base]

def to_rgba(color, alpha):
    color = color.strip()

    if color.startswith("rgb("):
        rgb = color.replace("rgb(", "").replace(")", "")
        return f"rgba({rgb}, {alpha})"

    if color.startswith("#"):
        color = color.lstrip("#")
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"

    return color

def darken_rgba(color, factor=0.6, alpha=0.4):
    r, g, b = map(int, re.findall(r"\d+", color)[:3])

    r = round(r * factor)
    g = round(g * factor)
    b = round(b * factor)

    return f"rgba({r}, {g}, {b}, {alpha})"

def add_line_traces(fig, df, columns=None, 
                    name_fn=str, legendgroup=None, opacity=1,
                    line_width=1.5, showlegend=True,
                    color_loops=1, colorscale=None, **kwargs):
    """Add multiple columns of a DataFrame as line traces.

    Works for deterministic forecasts (columns = leadtimes) and
    ensemble forecasts (columns = members).

    Parameters
    ----------
    fig : go.Figure
    df : pd.DataFrame
        Index is the x-axis (e.g. event_datetime), columns are the series.
    columns : list-like, optional
        Subset of df.columns to plot. Default: all columns.
    name_fn : callable
        Maps a column value to a legend name. Default: ``str``.
    legendgroup : str, optional
        If set, all traces share this legendgroup.
    """
    cols = columns if columns is not None else df.columns

    sample_scale = colorscale if colorscale is not None else default_color_scale
    colors = list(
        color_sampler(
            len(cols),
            colors=sample_scale,
            color_loops=color_loops,
        )
    )

    for col, color in zip(cols, colors):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=name_fn(col),
                opacity=opacity,
                line=dict(width=line_width, color=color),
                legendgroup=legendgroup,
                showlegend=showlegend,
                **kwargs,
            )
        )
    return fig

def add_observed_trace(fig, obs, name="obs", color="black", width=3,
                       dash="dash", showlegend=True, **kwargs):
    """Add an observed time-series trace."""
    if isinstance(obs, pd.DataFrame):
        y = obs.iloc[:, 0]
    else:
        y = obs

    fig.add_trace(
        go.Scatter(
            x=obs.index,
            y=y,
            mode="lines",
            name=name,
            line=dict(color=color, width=width, dash=dash),
            showlegend=showlegend,
            **kwargs,
        )
    )
    return fig

def add_shaded_quantiles(fig, df, color, name=None,
                         alpha_range=(0.1, 0.6), saturation_range=(0.5, 0.95),
                         show_median=True, median_width=2, median_dash='dash',
                         plot_function=go.Scatter, **kwargs):
    """Add shaded quantile bands to a figure.

    Parameters
    ----------
    df : pd.DataFrame
        Columns must include the quantile levels referenced in *bands*
        (and 0.5 when *show_median* is True).
    bands : sequence of (lo, hi) tuples
        Each pair defines a shaded region between quantiles ``df[lo]``
        and ``df[hi]``.  Ordered from widest to narrowest.
    alphas : sequence of float
        Fill opacity for each band (same length as *bands*).
    name : str, optional
        Legend / legendgroup label for the median line.
    """
    name_str = str(name) if name is not None else ""

    band_n = df.shape[1]
    bands = [(df.columns[i], df.columns[band_n-i-1]) for i in range(band_n//2)]
    alphas = np.linspace(alpha_range[0], alpha_range[1], len(bands))
    saturations = np.linspace(saturation_range[0], saturation_range[1], len(bands))

    for (lo, hi), alpha, saturation in zip(bands, alphas, saturations):
        
        fig.add_trace(
            plot_function(
                x=df.index,
                y=df[lo],
                mode="lines",
                line=dict(width=1, color=color),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            plot_function(
                x=df.index,
                y=df[hi],
                mode="lines",
                line=dict(width=1, color=color),
                fill="tonexty",
                fillcolor=darken_rgba(color, saturation, alpha),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if show_median and 0.5 in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[0.5],
                mode="lines",
                name=name_str,
                showlegend=False,
                line=dict(color=color, width=median_width, dash=median_dash),
                **kwargs,
            )
        )
    return fig

def apply_default_layout(fig, yaxis_title="", xaxis_title="",
                         layout_overrides=None, show_range_selector=True):
    """Apply default layout and optionally range-selector x-axes."""
    overrides = layout_overrides or {}
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        **update_layout,
        **overrides,
    )
    if show_range_selector:
        fig.update_xaxes(**update_xaxes)
    return fig

def plot_pd_deterministic(fig, df_, production_datetimes=None, **kwargs):
    '''
    Plots deterministic data by production date
    '''

    df = df_.copy()

    if production_datetimes is not None:
        df = df.loc[df.index.get_level_values('production_datetime').isin(production_datetimes), :]

    df = df.droplevel('leadtime').unstack('production_datetime')
    add_line_traces(fig, df, columns=df.columns, **kwargs)

def plot_lt_deterministic(fig, df_, leadtimes=None, **kwargs):
    '''
    Plots deterministic data by leadtime 
    '''

    df = df_.copy()

    if leadtimes is not None:
        df = df.loc[df.index.get_level_values('leadtime').isin(leadtimes), :]

    df = df.droplevel('production_datetime').unstack('leadtime')
    add_line_traces(fig, df, columns=df.columns, **kwargs)

def plot_pd_probabilistic(fig, df, production_datetimes=None, **kwargs):
    '''
    Plots probabilistic data by production date
    '''
    df_ = df.copy()

    if production_datetimes is not None:
        df_ = df_.loc[df_.index.get_level_values('production_datetime').isin(production_datetimes), :]

    colors_alias = kwargs.pop('colors', None)
    colorscale = kwargs.pop('colorscale', colors_alias)
    color_loops = kwargs.pop('color_loops', 1)

    production_values = df_.index.get_level_values('production_datetime').unique()
    colors = list(color_sampler(len(production_values), colors=colorscale, color_loops=color_loops))

    for pdt0, color in zip(production_values, colors):
        tmp = df_.xs(pdt0, level='production_datetime')

        if isinstance(tmp, pd.Series):
            tmp = tmp.to_frame(name='Q')

        if 'non_exceedance' in tmp.index.names:
            val_col = 'Q' if 'Q' in tmp.columns else tmp.columns[0]
            tmp = (
                tmp[val_col]
                .unstack('non_exceedance')
                .sort_index(axis=1)
            )
        else:
            tmp = tmp.sort_index(axis=1)

        if 'leadtime' in tmp.index.names:
            tmp = tmp.droplevel('leadtime')

        tmp = tmp.dropna(how='all')

        add_shaded_quantiles(
            fig,
            tmp,
            color=color,
            name=pdt0,
            **kwargs,
        )

    return fig

def plot_lt_probabilistic(fig, df, leadtimes=None, bands=None, **kwargs):
    '''
    Plots probabilistic data by leadtime 
    '''
    df_ = df.copy()

    if leadtimes is not None:
        df_ = df_.loc[df_.index.get_level_values('leadtime').isin(leadtimes), :]

    colors_alias = kwargs.pop('colors', None)
    colorscale = kwargs.pop('colorscale', colors_alias)
    color_loops = kwargs.pop('color_loops', 1)

    leadtime_values = df_.index.get_level_values('leadtime').unique()[::-1]
    colors = list(color_sampler(len(leadtime_values), colors=colorscale, color_loops=color_loops))

    for lt0, color in zip(leadtime_values, colors):
        tmp = df_.xs(lt0, level='leadtime')

        if isinstance(tmp, pd.Series):
            tmp = tmp.to_frame(name='Q')

        if 'non_exceedance' in tmp.index.names:
            val_col = 'Q' if 'Q' in tmp.columns else tmp.columns[0]
            tmp = (
                tmp[val_col]
                .unstack('non_exceedance')
                .sort_index(axis=1)
            )
        else:
            tmp = tmp.sort_index(axis=1)

        if bands is not None:
            selected_cols = [c for c in tmp.columns if c in bands]
            if selected_cols:
                tmp = tmp[selected_cols]

        if 'production_datetime' in tmp.index.names:
            tmp = tmp.droplevel('production_datetime')

        tmp = tmp.dropna(how='all')

        add_shaded_quantiles(
            fig,
            tmp,
            color=color,
            name=lt0,
            **kwargs,
        )

    return fig

def plot_pd_ensemble(fig, df, production_dates=None, ensembles=None, **kwargs):
    '''
    Plots ensemble data by production date
    '''
    df_ = df.copy()

    production_dates = kwargs.pop('production_datetimes', production_dates)

    col_names = list(df_.columns.names) if isinstance(df_.columns, pd.MultiIndex) else []

    if production_dates is not None:
        df_ = df_.loc[df_.index.get_level_values('production_datetime').isin(production_dates), :]

    if ensembles is not None:
        df_ = df_.loc[df_.index.get_level_values('ensemble_member').isin(ensembles), :]

    if 'leadtime' in df_.index.names:
        df_ = df_.droplevel('leadtime')

    df_ = df_.unstack(['production_datetime', 'ensemble_member'])
    df_ = df_.dropna(how='all')

    colors_alias = kwargs.pop('colors', None)
    colorscale = kwargs.pop('colorscale', colors_alias)
    color_loops = kwargs.pop('color_loops', 1)

    add_line_traces(
        fig,
        df_,
        columns=df_.columns,
        showlegend=False,
        colorscale=colorscale,
        color_loops=color_loops,
        **kwargs,
    )
    return fig
