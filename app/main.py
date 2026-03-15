"""RiskLab — Monte Carlo Portfolio Simulator (Streamlit entry point)."""
from __future__ import annotations

import sys
import os

# Ensure the app directory is on the path for sibling imports
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st

from data_loader import fetch_historical_data, compute_returns, compute_annualized_stats
from optimizer import generate_efficient_frontier
from simulator import run_simulation
from charts import (
    plot_cumulative_returns,
    plot_correlation_heatmap,
    plot_efficient_frontier,
    plot_simulation_paths,
    plot_terminal_distribution,
    plot_drawdown_distribution,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RiskLab — Monte Carlo Portfolio Simulator",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("RiskLab — Monte Carlo Portfolio Simulator")
st.caption(
    "Quantitative portfolio risk analysis using C++ GBM simulation, "
    "Modern Portfolio Theory, and interactive Plotly charts."
)

# ── Cached data fetching ──────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers_key: str, period: str):
    tickers = [t.strip().upper() for t in tickers_key.split(",") if t.strip()]
    prices = fetch_historical_data(tickers, period=period)
    returns = compute_returns(prices)
    stats = compute_annualized_stats(returns)
    return prices, returns, stats


@st.cache_data(ttl=3600, show_spinner=False)
def load_frontier(tickers_key: str, period: str, num_portfolios: int, rfr: float):
    _, returns, _ = load_data(tickers_key, period)
    return generate_efficient_frontier(returns, num_portfolios=num_portfolios, risk_free_rate=rfr)


# ── Sidebar — Inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    # Tickers
    ticker_input = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, GOOGL, JPM, GS",
    )
    tickers_raw = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    data_period = st.selectbox(
        "Historical data period",
        options=["1y", "2y", "3y", "5y", "10y"],
        index=3,
    )

    st.divider()

    # Simulation parameters
    num_sims = st.slider("Number of simulations", 1_000, 50_000, 10_000, step=1_000)
    horizon_months = st.slider("Time horizon (months)", 6, 60, 12)
    horizon_days = horizon_months * 21

    initial_value = st.number_input(
        "Initial portfolio value ($)", min_value=1_000, value=100_000, step=1_000
    )
    risk_free_rate = st.number_input(
        "Risk-free rate (annual)", min_value=0.0, max_value=0.20, value=0.05, step=0.005,
        format="%.3f"
    )

    st.divider()

    # Weight allocation
    weight_mode = st.radio(
        "Weight allocation",
        ["Manual", "Max Sharpe (from optimizer)", "Equal weight"],
    )

    use_fixed_seed = st.checkbox("Use fixed random seed (reproducible)", value=True)
    seed_val = st.number_input("Seed", min_value=0, value=42, step=1) if use_fixed_seed else None

    st.divider()
    run_button = st.button("Run Simulation", type="primary", use_container_width=True)


# ── Load historical data ──────────────────────────────────────────────────────
tickers_key = ", ".join(tickers_raw)

try:
    with st.spinner("Fetching historical data…"):
        prices_df, returns_df, asset_stats = load_data(tickers_key, data_period)
except Exception as exc:
    st.error(f"Data loading failed: {exc}")
    st.stop()

# Keep only tickers that were successfully downloaded
available_tickers = [t for t in tickers_raw if t in returns_df.columns]
if not available_tickers:
    st.error("No valid tickers found. Please check your input.")
    st.stop()
if len(available_tickers) < len(tickers_raw):
    dropped = set(tickers_raw) - set(available_tickers)
    st.warning(f"Could not download data for: {', '.join(dropped)}. Proceeding without them.")

prices_df  = prices_df[available_tickers]
returns_df = returns_df[available_tickers]


# ── Efficient frontier (needed for weight modes) ──────────────────────────────
try:
    with st.spinner("Computing efficient frontier…"):
        frontier_df, max_sharpe_port, min_vol_port = load_frontier(
            ", ".join(available_tickers), data_period, 5_000, risk_free_rate
        )
except Exception as exc:
    st.error(f"Efficient frontier computation failed: {exc}")
    st.stop()


# ── Determine weights ─────────────────────────────────────────────────────────
n = len(available_tickers)

if weight_mode == "Max Sharpe (from optimizer)":
    weights = max_sharpe_port["weights"]
    st.sidebar.success(
        "Using max Sharpe weights:\n"
        + "\n".join(f"  {t}: {w*100:.1f}%" for t, w in zip(available_tickers, weights))
    )
elif weight_mode == "Equal weight":
    weights = [1.0 / n] * n
else:
    # Manual sliders
    st.sidebar.subheader("Manual Weights")
    raw_weights = []
    for t in available_tickers:
        w = st.sidebar.slider(f"{t}", 0.0, 1.0, 1.0 / n, 0.01)
        raw_weights.append(w)
    total = sum(raw_weights)
    if abs(total - 1.0) > 0.01:
        st.sidebar.warning(f"Weights sum to {total:.2f} — will be normalized to 1.0")
    weights = [w / total for w in raw_weights] if total > 0 else [1.0 / n] * n


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Historical Context
# ═══════════════════════════════════════════════════════════════════════════════
st.header("1. Historical Context")

col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(plot_cumulative_returns(prices_df), use_container_width=True)

with col2:
    st.plotly_chart(plot_correlation_heatmap(returns_df), use_container_width=True)

# Annualized stats table
stats_table = pd.DataFrame(asset_stats).T
stats_table.columns = ["Annual Return", "Annual Volatility"]
stats_table = stats_table[stats_table.index.isin(available_tickers)]
stats_table["Annual Return"] = stats_table["Annual Return"].map("{:.2%}".format)
stats_table["Annual Volatility"] = stats_table["Annual Volatility"].map("{:.2%}".format)
st.dataframe(stats_table, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Efficient Frontier
# ═══════════════════════════════════════════════════════════════════════════════
st.header("2. Efficient Frontier")

st.plotly_chart(
    plot_efficient_frontier(frontier_df, max_sharpe_port, min_vol_port),
    use_container_width=True,
)

col_ms, col_mv = st.columns(2)
with col_ms:
    st.subheader("Max Sharpe Portfolio")
    ms_df = pd.DataFrame({
        "Ticker": available_tickers,
        "Weight": [f"{w*100:.1f}%" for w in max_sharpe_port["weights"]],
    })
    st.dataframe(ms_df, hide_index=True, use_container_width=True)
    st.metric("Sharpe Ratio", f"{max_sharpe_port['sharpe']:.3f}")
    st.metric("Expected Return", f"{max_sharpe_port['return']*100:.2f}%")
    st.metric("Volatility", f"{max_sharpe_port['volatility']*100:.2f}%")

with col_mv:
    st.subheader("Min Volatility Portfolio")
    mv_df = pd.DataFrame({
        "Ticker": available_tickers,
        "Weight": [f"{w*100:.1f}%" for w in min_vol_port["weights"]],
    })
    st.dataframe(mv_df, hide_index=True, use_container_width=True)
    st.metric("Sharpe Ratio", f"{min_vol_port['sharpe']:.3f}")
    st.metric("Expected Return", f"{min_vol_port['return']*100:.2f}%")
    st.metric("Volatility", f"{min_vol_port['volatility']*100:.2f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 & 4: Monte Carlo Simulation + Risk Summary
# ═══════════════════════════════════════════════════════════════════════════════
st.header("3. Monte Carlo Simulation")

if run_button or st.session_state.get("sim_results"):
    if run_button:
        try:
            with st.spinner(f"Running {num_sims:,} simulations × {horizon_days} days…"):
                sim_results = run_simulation(
                    tickers=available_tickers,
                    weights=weights,
                    num_simulations=num_sims,
                    horizon_days=horizon_days,
                    initial_value=float(initial_value),
                    returns_df=returns_df,
                    risk_free_rate=risk_free_rate,
                    seed=seed_val,
                )
            st.session_state["sim_results"] = sim_results
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")
            st.stop()

    sim_results = st.session_state["sim_results"]

    # Simulation fan chart
    st.plotly_chart(
        plot_simulation_paths(sim_results["paths"], float(initial_value), num_display=300),
        use_container_width=True,
    )

    col_hist, col_dd = st.columns(2)
    with col_hist:
        st.plotly_chart(
            plot_terminal_distribution(
                sim_results["terminal_values"],
                sim_results["var_95"],
                sim_results["var_99"],
                sim_results["cvar_95"],
                float(initial_value),
            ),
            use_container_width=True,
        )
    with col_dd:
        st.plotly_chart(
            plot_drawdown_distribution(sim_results["max_drawdowns"]),
            use_container_width=True,
        )

    # ── Section 4: Risk Summary ───────────────────────────────────────────────
    st.header("4. Risk Summary")

    iv = float(initial_value)
    var95 = sim_results["var_95"]
    var99 = sim_results["var_99"]
    cvar95 = sim_results["cvar_95"]

    row1 = st.columns(4)
    row2 = st.columns(4)

    row1[0].metric(
        "Expected Return (ann.)",
        f"{sim_results['expected_return']*100:.2f}%",
    )
    row1[1].metric(
        "Volatility (ann.)",
        f"{sim_results['volatility']*100:.2f}%",
    )
    row1[2].metric("Sharpe Ratio", f"{sim_results['sharpe']:.3f}")
    row1[3].metric(
        "Prob. of Loss",
        f"{sim_results['prob_of_loss']*100:.1f}%",
    )

    row2[0].metric(
        "VaR 95%",
        f"${var95:,.0f}",
        delta=f"-{var95/iv*100:.1f}%",
        delta_color="inverse",
    )
    row2[1].metric(
        "VaR 99%",
        f"${var99:,.0f}",
        delta=f"-{var99/iv*100:.1f}%",
        delta_color="inverse",
    )
    row2[2].metric(
        "CVaR 95%",
        f"${cvar95:,.0f}",
        delta=f"-{cvar95/iv*100:.1f}%",
        delta_color="inverse",
    )
    row2[3].metric(
        "Median Terminal Value",
        f"${sim_results['median_terminal_value']:,.0f}",
        delta=f"{(sim_results['median_terminal_value']-iv)/iv*100:.1f}%",
    )

    # Current weight display
    with st.expander("Current portfolio weights"):
        w_df = pd.DataFrame({
            "Ticker": sim_results["tickers"],
            "Weight": [f"{w*100:.2f}%" for w in sim_results["weights"]],
        })
        st.dataframe(w_df, hide_index=True, use_container_width=True)

else:
    st.info("Configure your portfolio in the sidebar and click **Run Simulation** to begin.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "RiskLab · Monte Carlo simulation engine written in C++17 with pybind11 bindings · "
    "Financial data provided by Yahoo Finance via yfinance · "
    "This tool is for educational and research purposes only."
)
