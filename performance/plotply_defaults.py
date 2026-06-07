import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re

default_color_scale = px.colors.sequential.Plasma

def color_sampler(n, color=None, loops=1):
    if color is None:
        color = default_color_scale

    if n <= 0:
        return

    n_base = max(1, round(n / loops))

    positions = [0.5] if n_base == 1 else [i / (n_base - 1) for i in range(n_base)]
    base_colors = px.colors.sample_colorscale(color, positions)

    for i in range(n):
        yield base_colors[i % n_base]


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
    width=1500,
    height=550,
    hovermode="x unified",
    margin=dict(l=60, r=30, t=40, b=60),
)

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


def add_line_traces(fig, df, columns=None, colors=None, colorscale=None,
                    name_fn=str, legendgroup=None, opacity=0.75,
                    line_width=1.5, showlegend=True, **kwargs):
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
    colors : list-like, optional
        Explicit colors per column. Overrides *colorscale*.
    colorscale : list-like, optional
        Plotly colorscale to sample from. Default: Plasma.
    name_fn : callable
        Maps a column value to a legend name. Default: ``str``.
    legendgroup : str, optional
        If set, all traces share this legendgroup.
    """
    cols = columns if columns is not None else df.columns
    if colors is None:
        colors = list(color_sampler(len(cols), colorscale))

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

def darken_rgba(color, factor=0.6, alpha=0.4):
    r, g, b = map(int, re.findall(r"\d+", color)[:3])

    r = round(r * factor)
    g = round(g * factor)
    b = round(b * factor)

    return f"rgba({r}, {g}, {b}, {alpha})"

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