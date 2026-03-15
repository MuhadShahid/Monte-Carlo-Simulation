"""Python-side tests for the C++ risklab_engine module."""
import math
import sys
import os
import pytest

# Allow running from project root with: python -m pytest tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import risklab_engine as rl


# ── RandomEngine ──────────────────────────────────────────────────────────────

class TestRandomEngine:
    def test_reproducibility(self):
        rng1 = rl.RandomEngine(42)
        rng2 = rl.RandomEngine(42)
        vals1 = [rng1.normal() for _ in range(100)]
        vals2 = [rng2.normal() for _ in range(100)]
        assert vals1 == vals2

    def test_different_seeds_differ(self):
        rng1 = rl.RandomEngine(1)
        rng2 = rl.RandomEngine(2)
        assert rng1.normal() != rng2.normal()

    def test_normal_vector_length(self):
        rng = rl.RandomEngine(0)
        v = rng.normal_vector(10)
        assert len(v) == 10

    def test_normal_distribution_properties(self):
        """Mean ≈ 0, std ≈ 1 for large N."""
        rng = rl.RandomEngine(123)
        n = 100_000
        vals = rng.normal_vector(n)
        mean = sum(vals) / n
        var = sum((x - mean) ** 2 for x in vals) / (n - 1)
        assert abs(mean) < 0.01, f"Mean too far from 0: {mean}"
        assert abs(math.sqrt(var) - 1.0) < 0.01, f"Std too far from 1: {math.sqrt(var)}"

    def test_correlated_normals_shape(self):
        rng = rl.RandomEngine(7)
        L = [[1.0, 0.0], [0.5, math.sqrt(0.75)]]
        result = rng.correlated_normals(L)
        assert len(result) == 2

    def test_reseed(self):
        rng = rl.RandomEngine(99)
        v1 = rng.normal()
        rng.reseed(99)
        v2 = rng.normal()
        assert v1 == v2


# ── Statistics ────────────────────────────────────────────────────────────────

class TestStatistics:
    def test_log_returns_basic(self):
        prices = [100.0, 110.0, 105.0]
        returns = rl.compute_log_returns(prices)
        assert len(returns) == 2
        assert abs(returns[0] - math.log(110.0 / 100.0)) < 1e-12
        assert abs(returns[1] - math.log(105.0 / 110.0)) < 1e-12

    def test_log_returns_too_short(self):
        with pytest.raises(Exception):
            rl.compute_log_returns([100.0])

    def test_covariance_matrix_diagonal(self):
        """Single asset: covariance matrix should be 1x1 with variance."""
        import math
        r = [0.01, -0.02, 0.015, -0.005, 0.02]
        cov = rl.compute_covariance_matrix([r])
        assert len(cov) == 1
        assert len(cov[0]) == 1
        # Compute expected variance manually
        n = len(r)
        mean = sum(r) / n
        expected_var = sum((x - mean) ** 2 for x in r) / (n - 1)
        assert abs(cov[0][0] - expected_var) < 1e-12

    def test_covariance_matrix_symmetry(self):
        r1 = [0.01, -0.02, 0.03, 0.005, -0.01]
        r2 = [0.02, -0.01, 0.025, 0.01, -0.005]
        cov = rl.compute_covariance_matrix([r1, r2])
        assert abs(cov[0][1] - cov[1][0]) < 1e-12

    def test_cholesky_2x2(self):
        """Test Cholesky of known 2x2 matrix."""
        # [[4, 2], [2, 3]] → L = [[2, 0], [1, sqrt(2)]]
        cov = [[4.0, 2.0], [2.0, 3.0]]
        L = rl.cholesky_decompose(cov)
        assert abs(L[0][0] - 2.0) < 1e-10
        assert abs(L[1][0] - 1.0) < 1e-10
        assert abs(L[1][1] - math.sqrt(2.0)) < 1e-10
        assert abs(L[0][1]) < 1e-10  # upper triangle = 0

    def test_cholesky_reconstruction(self):
        """L * L^T should equal original matrix."""
        cov = [[4.0, 2.0, 1.0], [2.0, 3.0, 0.5], [1.0, 0.5, 2.0]]
        L = rl.cholesky_decompose(cov)
        n = len(L)
        # Reconstruct A = L * L^T
        for i in range(n):
            for j in range(n):
                llt = sum(L[i][k] * L[j][k] for k in range(n))
                assert abs(llt - cov[i][j]) < 1e-10, f"Mismatch at [{i},{j}]"

    def test_cholesky_not_positive_definite(self):
        with pytest.raises(Exception):
            rl.cholesky_decompose([[1.0, 2.0], [2.0, 1.0]])

    def test_annualized_return(self):
        daily = [0.001] * 252
        assert abs(rl.annualized_return(daily) - 0.252) < 1e-10

    def test_annualized_volatility(self):
        # Constant returns → zero volatility
        daily = [0.01] * 100
        assert rl.annualized_volatility(daily) < 1e-10


# ── Portfolio ─────────────────────────────────────────────────────────────────

class TestPortfolio:
    def _make_two_asset_portfolio(self, w1=0.6, w2=0.4):
        p = rl.Portfolio()
        a1 = rl.Asset()
        a1.ticker = "A"
        a1.annualized_return = 0.10
        a1.annualized_volatility = 0.20
        p.add_asset(a1)

        a2 = rl.Asset()
        a2.ticker = "B"
        a2.annualized_return = 0.15
        a2.annualized_volatility = 0.25
        p.add_asset(a2)

        p.set_weights([w1, w2])
        # Daily covariance matrix (annual vol^2 / 252)
        var1 = (0.20 ** 2) / 252
        var2 = (0.25 ** 2) / 252
        cov12 = 0.5 * 0.20 * 0.25 / 252  # correlation = 0.5
        p.set_covariance_matrix([[var1, cov12], [cov12, var2]])
        return p

    def test_expected_return(self):
        p = self._make_two_asset_portfolio()
        expected = 0.6 * 0.10 + 0.4 * 0.15
        assert abs(p.expected_return() - expected) < 1e-12

    def test_volatility(self):
        p = self._make_two_asset_portfolio()
        w = [0.6, 0.4]
        var1 = (0.20 ** 2) / 252
        var2 = (0.25 ** 2) / 252
        cov12 = 0.5 * 0.20 * 0.25 / 252
        port_variance_daily = (
            w[0]**2 * var1
            + w[1]**2 * var2
            + 2 * w[0] * w[1] * cov12
        )
        expected_vol = math.sqrt(port_variance_daily * 252)
        assert abs(p.volatility() - expected_vol) < 1e-10

    def test_sharpe_ratio(self):
        p = self._make_two_asset_portfolio()
        rf = 0.05
        expected = (p.expected_return() - rf) / p.volatility()
        assert abs(p.sharpe_ratio(rf) - expected) < 1e-10

    def test_num_assets(self):
        p = self._make_two_asset_portfolio()
        assert p.num_assets() == 2


# ── Monte Carlo Simulator ─────────────────────────────────────────────────────

class TestMonteCarlo:
    def _make_single_asset_portfolio(self, mu=0.10, sigma=0.20):
        p = rl.Portfolio()
        a = rl.Asset()
        a.ticker = "X"
        a.annualized_return = mu
        a.annualized_volatility = sigma
        p.add_asset(a)
        p.set_weights([1.0])
        p.set_covariance_matrix([[(sigma ** 2) / 252]])
        return p

    def test_output_shape(self):
        p = self._make_single_asset_portfolio()
        rng = rl.RandomEngine(42)
        sim = rl.MonteCarloSimulator(100, 252, p, rng, 1.0)
        result = sim.run()
        assert len(result.paths) == 100
        assert all(len(path) == 253 for path in result.paths)  # 252 steps + initial
        assert len(result.terminal_values) == 100

    def test_initial_value(self):
        p = self._make_single_asset_portfolio()
        rng = rl.RandomEngine(42)
        sim = rl.MonteCarloSimulator(50, 10, p, rng, 12345.0)
        result = sim.run()
        assert all(abs(path[0] - 12345.0) < 1e-10 for path in result.paths)

    def test_terminal_values_match_paths(self):
        p = self._make_single_asset_portfolio()
        rng = rl.RandomEngine(42)
        sim = rl.MonteCarloSimulator(50, 20, p, rng, 1.0)
        result = sim.run()
        for i, (path, tv) in enumerate(zip(result.paths, result.terminal_values)):
            assert abs(path[-1] - tv) < 1e-10, f"Path/terminal mismatch at sim {i}"

    def test_mean_convergence(self):
        """
        Mean terminal value should converge to exp(mu * T) as num_sims grows.
        GBM: E[S_T] = S_0 * exp(mu * T)
        """
        mu = 0.10
        T_years = 1.0
        p = self._make_single_asset_portfolio(mu=mu, sigma=0.20)
        rng = rl.RandomEngine(0)
        sim = rl.MonteCarloSimulator(20_000, 252, p, rng, 1.0)
        result = sim.run()
        mean_tv = sum(result.terminal_values) / len(result.terminal_values)
        expected_mean = math.exp(mu * T_years)
        # Allow 2% error with 20k paths
        assert abs(mean_tv - expected_mean) / expected_mean < 0.02, (
            f"Mean {mean_tv:.4f} vs expected {expected_mean:.4f}"
        )


# ── Risk Metrics ──────────────────────────────────────────────────────────────

class TestRiskMetrics:
    def _uniform_terminal_values(self):
        """100 values from 90 to 109 (initial = 100)."""
        return [float(90 + i) for i in range(20)]  # 90..109

    def test_var_95(self):
        # With initial=100, 5th percentile of losses
        tv = list(range(50, 150))  # 100 values; 5th percentile = 54
        var = rl.value_at_risk(tv, 0.95, 100.0)
        assert var > 0  # must be a loss

    def test_var_ordering(self):
        tv = list(range(50, 150))
        var95 = rl.value_at_risk(tv, 0.95, 100.0)
        var99 = rl.value_at_risk(tv, 0.99, 100.0)
        assert var99 >= var95, "99% VaR should be >= 95% VaR"

    def test_cvar_gte_var(self):
        tv = list(range(50, 150))
        var95  = rl.value_at_risk(tv, 0.95, 100.0)
        cvar95 = rl.conditional_var(tv, 0.95, 100.0)
        assert cvar95 >= var95, "CVaR should be >= VaR at same confidence"

    def test_max_drawdown_no_drawdown(self):
        path = [1.0, 1.1, 1.2, 1.3, 1.4]
        dd = rl.max_drawdown(path)
        assert abs(dd) < 1e-10

    def test_max_drawdown_full_loss(self):
        path = [1.0, 0.5]
        dd = rl.max_drawdown(path)
        assert abs(dd - 0.5) < 1e-10

    def test_max_drawdown_recovery(self):
        path = [1.0, 0.8, 0.6, 0.9, 1.1]
        dd = rl.max_drawdown(path)
        # Peak = 1.0, trough = 0.6 → drawdown = 0.4
        assert abs(dd - 0.4) < 1e-10

    def test_sharpe_from_paths_positive(self):
        # Steadily growing paths → positive Sharpe
        paths = [[1.0 * (1.001 ** t) for t in range(253)] for _ in range(100)]
        sharpe = rl.sharpe_from_paths(paths, 0.05)
        assert sharpe > 0
