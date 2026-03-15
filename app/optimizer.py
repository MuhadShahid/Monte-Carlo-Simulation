"""Efficient frontier computation via random portfolio sampling using the C++ engine."""
from __future__ import annotations

import numpy as np
import pandas as pd

import risklab_engine as rl


def generate_efficient_frontier(
    returns: pd.DataFrame,
    num_portfolios: int = 10_000,
    risk_free_rate: float = 0.05,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Generate random portfolio weight combinations and compute return/risk/Sharpe.

    Parameters
    ----------
    returns:
        Daily log returns, columns = tickers.
    num_portfolios:
        Number of random portfolios to sample.
    risk_free_rate:
        Annual risk-free rate used for Sharpe calculation.

    Returns
    -------
    frontier_df:
        DataFrame with columns: weights, return, volatility, sharpe.
    max_sharpe:
        Dict with keys: weights, return, volatility, sharpe.
    min_vol:
        Dict with keys: weights, return, volatility, sharpe.
    """
    tickers = list(returns.columns)
    n = len(tickers)

    # Build covariance matrix from historical data (via C++ engine)
    returns_list = [returns[t].tolist() for t in tickers]
    cov_matrix = rl.compute_covariance_matrix(returns_list)

    # Annualized stats per asset
    ann_returns = {t: float(returns[t].mean() * 252) for t in tickers}
    ann_vols = {t: float(returns[t].std() * np.sqrt(252)) for t in tickers}

    records = []
    rng = np.random.default_rng(seed=42)

    for _ in range(num_portfolios):
        # Random weights that sum to 1
        w = rng.random(n)
        w /= w.sum()

        # Build C++ Portfolio object
        portfolio = rl.Portfolio()
        for i, t in enumerate(tickers):
            asset = rl.Asset()
            asset.ticker = t
            asset.annualized_return = ann_returns[t]
            asset.annualized_volatility = ann_vols[t]
            portfolio.add_asset(asset)
        portfolio.set_weights(w.tolist())
        portfolio.set_covariance_matrix(cov_matrix)

        p_ret = portfolio.expected_return()
        p_vol = portfolio.volatility()
        p_sharpe = portfolio.sharpe_ratio(risk_free_rate)

        records.append({
            "weights": w.tolist(),
            "return": p_ret,
            "volatility": p_vol,
            "sharpe": p_sharpe,
        })

    frontier_df = pd.DataFrame(records)

    # Identify special portfolios
    max_sharpe_idx = frontier_df["sharpe"].idxmax()
    min_vol_idx = frontier_df["volatility"].idxmin()

    def _extract(idx: int) -> dict:
        row = frontier_df.iloc[idx]
        return {
            "weights": row["weights"],
            "tickers": tickers,
            "return": row["return"],
            "volatility": row["volatility"],
            "sharpe": row["sharpe"],
        }

    return frontier_df, _extract(max_sharpe_idx), _extract(min_vol_idx)
