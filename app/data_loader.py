"""Data ingestion and preprocessing: fetches historical price data via yfinance."""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _forward_fill_and_drop(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill missing values, then drop any remaining NaN rows."""
    return df.ffill().dropna()


def fetch_historical_data(
    tickers: list[str],
    period: str = "5y",
) -> pd.DataFrame:
    """
    Download adjusted close prices for the given tickers.

    Parameters
    ----------
    tickers:
        List of ticker symbols (e.g. ["AAPL", "MSFT"]).
    period:
        yfinance period string (e.g. "5y", "2y", "1y").

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date, columns = tickers, values = adjusted close prices.
        Tickers that fail to download are excluded with a warning.
    """
    if not tickers:
        raise ValueError("Ticker list is empty")

    tickers = [t.strip().upper() for t in tickers]

    try:
        raw = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        raise RuntimeError(f"yfinance download failed: {exc}") from exc

    # yfinance returns MultiIndex columns when multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=0)
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw

    # If single ticker, yfinance may return a Series
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    # Ensure all requested tickers are present; warn about missing ones
    present = set(prices.columns)
    missing = [t for t in tickers if t not in present]
    if missing:
        logger.warning("Failed to download data for: %s", missing)

    # Drop tickers with all-NaN columns
    prices = prices.dropna(axis=1, how="all")

    if prices.empty:
        raise RuntimeError("No valid price data downloaded for any ticker")

    prices = _forward_fill_and_drop(prices)
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from a price DataFrame.

    Returns
    -------
    pd.DataFrame
        Log returns, same columns as input, one fewer row.
    """
    return np.log(prices / prices.shift(1)).dropna()


def compute_annualized_stats(returns: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Compute annualized mean return and volatility for each asset.

    Returns
    -------
    dict
        {ticker: {"return": float, "volatility": float}}
    """
    stats: dict[str, dict[str, float]] = {}
    for col in returns.columns:
        series = returns[col].dropna()
        annualized_ret = float(series.mean() * 252)
        annualized_vol = float(series.std() * np.sqrt(252))
        stats[col] = {"return": annualized_ret, "volatility": annualized_vol}
    return stats
