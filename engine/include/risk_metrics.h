#pragma once
#include <vector>

namespace risklab {

// Value at Risk: loss at given confidence level (e.g., 0.95, 0.99)
// Returns a positive loss amount. terminal_values are portfolio end values.
// initial_value is used to convert to PnL. VaR = initial_value - percentile(terminal_values, 1-confidence)
double value_at_risk(
    const std::vector<double>& terminal_values,
    double confidence,
    double initial_value = 1.0);

// Conditional VaR (Expected Shortfall): mean loss beyond the VaR threshold
double conditional_var(
    const std::vector<double>& terminal_values,
    double confidence,
    double initial_value = 1.0);

// Maximum peak-to-trough drawdown in a single path
double max_drawdown(const std::vector<double>& path);

// Annualized Sharpe ratio computed from simulated paths
// Computes daily returns across all paths, annualizes, subtracts risk-free rate
double sharpe_from_paths(
    const std::vector<std::vector<double>>& paths,
    double risk_free_rate = 0.05);

} // namespace risklab
