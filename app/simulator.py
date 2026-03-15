"""Orchestrates Monte Carlo simulation runs via C++ bindings."""
from __future__ import annotations

import numpy as np
import pandas as pd

import risklab_engine as rl


def run_simulation(
    tickers: list[str],
    weights: list[float],
    num_simulations: int,
    horizon_days: int,
    initial_value: float,
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.05,
    seed: int | None = 42,
) -> dict:
    """
    Run a Monte Carlo portfolio simulation using the C++ engine.

    Parameters
    ----------
    tickers:
        Asset tickers (must match columns in returns_df).
    weights:
        Portfolio weights (will be normalized to sum to 1).
    num_simulations:
        Number of simulation paths.
    horizon_days:
        Simulation horizon in trading days.
    initial_value:
        Starting portfolio value (e.g. 100_000).
    returns_df:
        Daily log returns, columns include all tickers.
    risk_free_rate:
        Annual risk-free rate.
    seed:
        RNG seed. Pass None for random seed.

    Returns
    -------
    dict with keys:
        paths, terminal_values, var_95, var_99, cvar_95, cvar_99,
        max_drawdowns, sharpe, expected_return, volatility,
        prob_of_loss, median_terminal_value
    """
    # Normalize weights
    w = np.array(weights, dtype=float)
    w /= w.sum()

    # Build per-asset stats from historical data
    returns_list = [returns_df[t].tolist() for t in tickers]
    cov_matrix = rl.compute_covariance_matrix(returns_list)

    ann_returns = {t: float(returns_df[t].mean() * 252) for t in tickers}
    ann_vols    = {t: float(returns_df[t].std() * np.sqrt(252)) for t in tickers}

    # Build C++ Portfolio
    portfolio = rl.Portfolio()
    for i, t in enumerate(tickers):
        asset = rl.Asset()
        asset.ticker = t
        asset.annualized_return = ann_returns[t]
        asset.annualized_volatility = ann_vols[t]
        portfolio.add_asset(asset)
    portfolio.set_weights(w.tolist())
    portfolio.set_covariance_matrix(cov_matrix)

    # RNG
    actual_seed = seed if seed is not None else np.random.randint(0, 2**32)
    rng = rl.RandomEngine(actual_seed)

    # Run simulation
    sim = rl.MonteCarloSimulator(
        num_simulations,
        horizon_days,
        portfolio,
        rng,
        initial_value,
    )
    result = sim.run()

    paths = result.paths
    terminal_values = result.terminal_values

    # Risk metrics
    var_95  = rl.value_at_risk(terminal_values, 0.95, initial_value)
    var_99  = rl.value_at_risk(terminal_values, 0.99, initial_value)
    cvar_95 = rl.conditional_var(terminal_values, 0.95, initial_value)
    cvar_99 = rl.conditional_var(terminal_values, 0.99, initial_value)

    # Max drawdown per path
    max_drawdowns = [rl.max_drawdown(p) for p in paths]

    # Sharpe from simulated paths
    sharpe = rl.sharpe_from_paths(paths, risk_free_rate)

    # Portfolio-level stats
    p_return = portfolio.expected_return()
    p_vol    = portfolio.volatility()

    # Probability of loss
    n_loss = sum(1 for v in terminal_values if v < initial_value)
    prob_of_loss = n_loss / len(terminal_values)

    median_terminal = float(np.median(terminal_values))

    return {
        "paths": paths,
        "terminal_values": terminal_values,
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "cvar_99": cvar_99,
        "max_drawdowns": max_drawdowns,
        "sharpe": sharpe,
        "expected_return": p_return,
        "volatility": p_vol,
        "prob_of_loss": prob_of_loss,
        "median_terminal_value": median_terminal,
        "weights": w.tolist(),
        "tickers": tickers,
        "initial_value": initial_value,
    }
