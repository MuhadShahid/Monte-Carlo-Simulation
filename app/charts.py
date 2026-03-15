"""All Plotly visualization functions for RiskLab. Returns go.Figure objects."""
from __future__ import annotations

import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

_THEME = "plotly_dark"
_ACCENT = "#00d4ff"
_ACCENT2 = "#ff6b35"
_ACCENT3 = "#7bed9f"


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    max_sharpe: dict,
    min_vol: dict,
) -> go.Figure:
    """Scatter plot of efficient frontier colored by Sharpe ratio."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=frontier_df["volatility"] * 100,
        y=frontier_df["return"] * 100,
        mode="markers",
        marker=dict(
            color=frontier_df["sharpe"],
            colorscale="Viridis",
            size=3,
            opacity=0.6,
            colorbar=dict(title="Sharpe Ratio"),
        ),
        name="Portfolios",
        hovertemplate=(
            "Volatility: %{x:.2f}%<br>"
            "Return: %{y:.2f}%<br>"
            "Sharpe: %{marker.color:.3f}<extra></extra>"
        ),
    ))

    # Max Sharpe point
    fig.add_trace(go.Scatter(
        x=[max_sharpe["volatility"] * 100],
        y=[max_sharpe["return"] * 100],
        mode="markers+text",
        marker=dict(color=_ACCENT2, size=14, symbol="star"),
        text=["Max Sharpe"],
        textposition="top center",
        name=f"Max Sharpe ({max_sharpe['sharpe']:.2f})",
    ))

    # Min volatility point
    fig.add_trace(go.Scatter(
        x=[min_vol["volatility"] * 100],
        y=[min_vol["return"] * 100],
        mode="markers+text",
        marker=dict(color=_ACCENT3, size=14, symbol="diamond"),
        text=["Min Vol"],
        textposition="top center",
        name=f"Min Volatility ({min_vol['volatility']*100:.1f}%)",
    ))

    fig.update_layout(
        template=_THEME,
        title="Efficient Frontier (Random Portfolio Sampling)",
        xaxis_title="Annual Volatility (%)",
        yaxis_title="Annual Return (%)",
        height=480,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig


def plot_simulation_paths(
    paths: list[list[float]],
    initial_value: float,
    num_display: int = 200,
) -> go.Figure:
    """Fan chart of Monte Carlo paths with percentile cone."""
    fig = go.Figure()

    # Sample paths to display
    display_paths = paths
    if len(paths) > num_display:
        display_paths = random.sample(paths, num_display)

    horizon = len(paths[0]) - 1
    x = list(range(horizon + 1))

    # Thin lines for individual paths
    for path in display_paths:
        fig.add_trace(go.Scatter(
            x=x,
            y=path,
            mode="lines",
            line=dict(color="rgba(0, 212, 255, 0.07)", width=0.5),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Compute percentile bands across all paths
    arr = np.array(paths)
    p5  = np.percentile(arr, 5, axis=0)
    p25 = np.percentile(arr, 25, axis=0)
    p50 = np.percentile(arr, 50, axis=0)
    p75 = np.percentile(arr, 75, axis=0)
    p95 = np.percentile(arr, 95, axis=0)

    # 5th–95th percentile shaded cone
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=p95.tolist() + p5.tolist()[::-1],
        fill="toself",
        fillcolor="rgba(0, 212, 255, 0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="5th–95th Pct",
        hoverinfo="skip",
    ))

    # 25th–75th percentile shaded cone
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=p75.tolist() + p25.tolist()[::-1],
        fill="toself",
        fillcolor="rgba(0, 212, 255, 0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="25th–75th Pct",
        hoverinfo="skip",
    ))

    # Median path
    fig.add_trace(go.Scatter(
        x=x,
        y=p50,
        mode="lines",
        line=dict(color=_ACCENT, width=2.5),
        name="Median Path",
    ))

    # Initial value reference line
    fig.add_hline(
        y=initial_value,
        line_dash="dash",
        line_color="rgba(255,255,255,0.3)",
        annotation_text="Initial Value",
        annotation_position="bottom right",
    )

    fig.update_layout(
        template=_THEME,
        title=f"Monte Carlo Simulation Paths ({len(paths):,} simulations)",
        xaxis_title="Trading Days",
        yaxis_title="Portfolio Value ($)",
        height=480,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig


def plot_terminal_distribution(
    terminal_values: list[float],
    var_95: float,
    var_99: float,
    cvar_95: float,
    initial_value: float,
) -> go.Figure:
    """Histogram of terminal portfolio values with VaR/CVaR markers."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=terminal_values,
        nbinsx=80,
        marker_color=_ACCENT,
        opacity=0.75,
        name="Terminal Values",
    ))

    # VaR and CVaR vertical lines
    var_95_val = initial_value - var_95
    var_99_val = initial_value - var_99
    cvar_95_val = initial_value - cvar_95

    for val, label, color in [
        (var_95_val,  "VaR 95%",  "#ff6b35"),
        (var_99_val,  "VaR 99%",  "#ff0000"),
        (cvar_95_val, "CVaR 95%", "#ff00ff"),
        (initial_value, "Initial Value", "rgba(255,255,255,0.5)"),
    ]:
        fig.add_vline(
            x=val,
            line_dash="dash" if val != initial_value else "dot",
            line_color=color,
            annotation_text=label,
            annotation_position="top right",
            annotation_font_color=color,
        )

    n_loss = sum(1 for v in terminal_values if v < initial_value)
    prob_loss = n_loss / len(terminal_values) * 100

    fig.update_layout(
        template=_THEME,
        title=f"Terminal Portfolio Value Distribution  |  P(loss) = {prob_loss:.1f}%",
        xaxis_title="Terminal Portfolio Value ($)",
        yaxis_title="Count",
        height=420,
        bargap=0.02,
    )
    return fig


def plot_drawdown_distribution(max_drawdowns: list[float]) -> go.Figure:
    """Histogram of maximum drawdowns across all simulation paths."""
    pct = [d * 100 for d in max_drawdowns]

    fig = go.Figure(go.Histogram(
        x=pct,
        nbinsx=60,
        marker_color=_ACCENT2,
        opacity=0.75,
        name="Max Drawdown",
    ))

    median_dd = float(np.median(pct))
    fig.add_vline(
        x=median_dd,
        line_dash="dash",
        line_color=_ACCENT,
        annotation_text=f"Median: {median_dd:.1f}%",
        annotation_position="top right",
    )

    fig.update_layout(
        template=_THEME,
        title="Maximum Drawdown Distribution",
        xaxis_title="Max Drawdown (%)",
        yaxis_title="Count",
        height=380,
        bargap=0.02,
    )
    return fig


def plot_correlation_heatmap(returns_df: pd.DataFrame) -> go.Figure:
    """Annotated heatmap of asset return correlations."""
    corr = returns_df.corr()
    tickers = list(corr.columns)

    annotations = []
    for i, row in enumerate(tickers):
        for j, col in enumerate(tickers):
            annotations.append(dict(
                x=col,
                y=row,
                text=f"{corr.loc[row, col]:.2f}",
                showarrow=False,
                font=dict(color="white" if abs(corr.loc[row, col]) < 0.7 else "black"),
            ))

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=tickers,
        y=tickers,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlation"),
    ))

    fig.update_layout(
        template=_THEME,
        title="Asset Return Correlation Matrix",
        annotations=annotations,
        height=400,
    )
    return fig


def plot_cumulative_returns(prices_df: pd.DataFrame) -> go.Figure:
    """Normalized cumulative return of each asset over the historical period."""
    normalized = prices_df / prices_df.iloc[0] * 100

    colors = px.colors.qualitative.Plotly
    fig = go.Figure()

    for i, col in enumerate(normalized.columns):
        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized[col],
            mode="lines",
            name=col,
            line=dict(color=colors[i % len(colors)], width=1.5),
        ))

    fig.add_hline(
        y=100,
        line_dash="dot",
        line_color="rgba(255,255,255,0.2)",
    )

    fig.update_layout(
        template=_THEME,
        title="Historical Cumulative Returns (Base = 100)",
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        height=380,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified",
    )
    return fig
