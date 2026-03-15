"""Tests for the data loading and preprocessing pipeline."""
import sys
import os
import math
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from data_loader import compute_returns, compute_annualized_stats


class TestComputeReturns:
    def _make_prices(self, values: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"A": values}, index=pd.date_range("2020-01-01", periods=len(values)))

    def test_output_length(self):
        prices = self._make_prices([100, 110, 105, 115])
        returns = compute_returns(prices)
        assert len(returns) == 3  # one fewer row

    def test_log_return_value(self):
        prices = self._make_prices([100.0, 110.0])
        returns = compute_returns(prices)
        expected = math.log(110.0 / 100.0)
        assert abs(returns["A"].iloc[0] - expected) < 1e-12

    def test_multicolumn(self):
        df = pd.DataFrame(
            {"A": [100, 110, 105], "B": [200, 210, 195]},
            index=pd.date_range("2020-01-01", periods=3),
        )
        returns = compute_returns(df)
        assert list(returns.columns) == ["A", "B"]
        assert len(returns) == 2


class TestComputeAnnualizedStats:
    def test_keys_present(self):
        returns = pd.DataFrame(
            {"A": [0.01, -0.005, 0.02, 0.003, -0.01]},
            index=pd.date_range("2020-01-01", periods=5),
        )
        stats = compute_annualized_stats(returns)
        assert "A" in stats
        assert "return" in stats["A"]
        assert "volatility" in stats["A"]

    def test_constant_returns_zero_vol(self):
        returns = pd.DataFrame(
            {"A": [0.001] * 100},
            index=pd.date_range("2020-01-01", periods=100),
        )
        stats = compute_annualized_stats(returns)
        assert stats["A"]["volatility"] < 1e-10

    def test_annualization_factor(self):
        """Daily return of 0.001 → annualized return = 0.001 * 252."""
        daily_ret = 0.001
        returns = pd.DataFrame(
            {"X": [daily_ret] * 252},
            index=pd.date_range("2020-01-01", periods=252),
        )
        stats = compute_annualized_stats(returns)
        assert abs(stats["X"]["return"] - daily_ret * 252) < 1e-10

    def test_multiple_tickers(self):
        returns = pd.DataFrame(
            {"A": [0.01] * 50, "B": [-0.005] * 50},
            index=pd.date_range("2020-01-01", periods=50),
        )
        stats = compute_annualized_stats(returns)
        assert set(stats.keys()) == {"A", "B"}
        assert stats["A"]["return"] > stats["B"]["return"]
