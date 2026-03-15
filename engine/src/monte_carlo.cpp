#include "monte_carlo.h"
#include "statistics.h"
#include <cmath>
#include <stdexcept>

namespace risklab {

MonteCarloSimulator::MonteCarloSimulator(
    int num_simulations,
    int horizon_days,
    const Portfolio& portfolio,
    RandomEngine& rng,
    double initial_value)
    : num_simulations_(num_simulations)
    , horizon_days_(horizon_days)
    , portfolio_(portfolio)
    , rng_(rng)
    , initial_value_(initial_value)
{}

SimulationResult MonteCarloSimulator::run() {
    std::size_t n = portfolio_.num_assets();
    if (n == 0) throw std::runtime_error("Portfolio has no assets");

    const auto& weights = portfolio_.weights();
    const auto& assets = portfolio_.assets();
    const auto& cov_matrix = portfolio_.covariance_matrix();

    if (weights.size() != n) throw std::runtime_error("Weights not set on portfolio");
    if (cov_matrix.empty()) throw std::runtime_error("Covariance matrix not set");

    // Pre-compute Cholesky factor (daily covariance)
    auto L = cholesky_decompose(cov_matrix);

    // Daily time step
    const double dt = 1.0 / 252.0;

    // Per-asset daily drift: (mu_i - 0.5*sigma_i^2)*dt
    // mu_i and sigma_i are annualized; convert to daily
    std::vector<double> daily_drift(n);
    for (std::size_t i = 0; i < n; ++i) {
        double mu = assets[i].annualized_return;
        double sigma = assets[i].annualized_volatility;
        daily_drift[i] = (mu - 0.5 * sigma * sigma) * dt;
        // Note: the diffusion component comes from correlated_normals(L) where
        // L is the Cholesky of the DAILY covariance matrix. That call already
        // produces draws with the correct per-step scale (sigma*sqrt(dt)), so
        // we must NOT apply any additional vol scaling to z[i].
    }

    SimulationResult result;
    result.paths.resize(num_simulations_);
    result.terminal_values.resize(num_simulations_);

    for (int sim = 0; sim < num_simulations_; ++sim) {
        auto& path = result.paths[sim];
        path.resize(horizon_days_ + 1);
        path[0] = initial_value_;

        // Track individual asset values (normalized to initial weights)
        std::vector<double> asset_values(n);
        for (std::size_t i = 0; i < n; ++i) {
            asset_values[i] = weights[i] * initial_value_;
        }

        for (int t = 0; t < horizon_days_; ++t) {
            // Generate correlated standard normal draws
            auto z = rng_.correlated_normals(L);

            // Apply GBM to each asset.
            // z[i] = (L * standard_normal)[i] already has covariance Sigma_daily,
            // i.e. z[i] ~ N(0, sigma_i^2 * dt). No additional vol scaling needed.
            for (std::size_t i = 0; i < n; ++i) {
                double shock = daily_drift[i] + z[i];
                asset_values[i] *= std::exp(shock);
            }

            // Portfolio value = sum of asset values
            double portfolio_value = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                portfolio_value += asset_values[i];
            }
            path[t + 1] = portfolio_value;
        }

        result.terminal_values[sim] = path[horizon_days_];
    }

    return result;
}

} // namespace risklab
