#include "risk_metrics.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

namespace risklab {

double value_at_risk(
    const std::vector<double>& terminal_values,
    double confidence,
    double initial_value)
{
    if (terminal_values.empty()) throw std::invalid_argument("No terminal values provided");
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("Confidence must be in (0, 1)");
    }

    std::vector<double> sorted = terminal_values;
    std::sort(sorted.begin(), sorted.end());

    // VaR at confidence level: loss at the (1 - confidence) percentile
    std::size_t idx = static_cast<std::size_t>((1.0 - confidence) * sorted.size());
    if (idx >= sorted.size()) idx = sorted.size() - 1;

    double threshold_value = sorted[idx];
    double loss = initial_value - threshold_value;
    return loss; // positive = loss, negative = gain
}

double conditional_var(
    const std::vector<double>& terminal_values,
    double confidence,
    double initial_value)
{
    if (terminal_values.empty()) throw std::invalid_argument("No terminal values provided");
    if (confidence <= 0.0 || confidence >= 1.0) {
        throw std::invalid_argument("Confidence must be in (0, 1)");
    }

    std::vector<double> sorted = terminal_values;
    std::sort(sorted.begin(), sorted.end());

    std::size_t cutoff = static_cast<std::size_t>((1.0 - confidence) * sorted.size());
    if (cutoff == 0) cutoff = 1;

    // CVaR = mean loss of worst (1 - confidence) fraction
    double sum = 0.0;
    for (std::size_t i = 0; i < cutoff; ++i) {
        sum += sorted[i];
    }
    double mean_tail_value = sum / cutoff;
    return initial_value - mean_tail_value;
}

double max_drawdown(const std::vector<double>& path) {
    if (path.empty()) return 0.0;
    double peak = path[0];
    double max_dd = 0.0;
    for (double v : path) {
        if (v > peak) peak = v;
        double dd = (peak - v) / peak;
        if (dd > max_dd) max_dd = dd;
    }
    return max_dd;
}

double sharpe_from_paths(
    const std::vector<std::vector<double>>& paths,
    double risk_free_rate)
{
    if (paths.empty()) return 0.0;

    // Collect all daily returns across all paths
    std::vector<double> all_returns;
    for (const auto& path : paths) {
        for (std::size_t t = 1; t < path.size(); ++t) {
            if (path[t - 1] > 0.0) {
                all_returns.push_back(std::log(path[t] / path[t - 1]));
            }
        }
    }

    if (all_returns.empty()) return 0.0;

    double mean = std::accumulate(all_returns.begin(), all_returns.end(), 0.0)
                  / all_returns.size();

    double variance = 0.0;
    for (double r : all_returns) {
        double d = r - mean;
        variance += d * d;
    }
    variance /= (all_returns.size() - 1);
    double daily_vol = std::sqrt(variance);

    if (daily_vol == 0.0) return 0.0;

    double annualized_mean = mean * 252.0;
    double annualized_vol = daily_vol * std::sqrt(252.0);

    return (annualized_mean - risk_free_rate) / annualized_vol;
}

} // namespace risklab
