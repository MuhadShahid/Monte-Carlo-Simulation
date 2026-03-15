#include "statistics.h"
#include <cmath>
#include <stdexcept>
#include <numeric>

namespace risklab {

std::vector<double> compute_log_returns(const std::vector<double>& prices) {
    if (prices.size() < 2) {
        throw std::invalid_argument("Need at least 2 prices to compute returns");
    }
    std::vector<double> returns;
    returns.reserve(prices.size() - 1);
    for (std::size_t i = 1; i < prices.size(); ++i) {
        if (prices[i - 1] <= 0.0 || prices[i] <= 0.0) {
            throw std::invalid_argument("Prices must be positive");
        }
        returns.push_back(std::log(prices[i] / prices[i - 1]));
    }
    return returns;
}

std::vector<std::vector<double>> compute_covariance_matrix(
    const std::vector<std::vector<double>>& returns)
{
    std::size_t n = returns.size();
    if (n == 0) throw std::invalid_argument("No return series provided");

    std::size_t T = returns[0].size();
    for (std::size_t i = 1; i < n; ++i) {
        if (returns[i].size() != T) {
            throw std::invalid_argument("All return series must have the same length");
        }
    }
    if (T < 2) throw std::invalid_argument("Need at least 2 observations");

    // Compute means
    std::vector<double> means(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        means[i] = std::accumulate(returns[i].begin(), returns[i].end(), 0.0) / T;
    }

    // Sample covariance: 1/(T-1) * sum((r_i - mu_i)(r_j - mu_j))
    std::vector<std::vector<double>> cov(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            double s = 0.0;
            for (std::size_t t = 0; t < T; ++t) {
                s += (returns[i][t] - means[i]) * (returns[j][t] - means[j]);
            }
            cov[i][j] = s / (T - 1);
            cov[j][i] = cov[i][j];
        }
    }
    return cov;
}

std::vector<std::vector<double>> cholesky_decompose(
    const std::vector<std::vector<double>>& cov_matrix)
{
    std::size_t n = cov_matrix.size();
    if (n == 0) throw std::invalid_argument("Matrix is empty");

    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < j; ++k) {
                s += L[i][k] * L[j][k];
            }

            if (i == j) {
                double diag = cov_matrix[i][i] - s;
                if (diag <= 0.0) {
                    throw std::runtime_error(
                        "Covariance matrix is not positive definite (Cholesky failed at index "
                        + std::to_string(i) + ")");
                }
                L[i][j] = std::sqrt(diag);
            } else {
                if (L[j][j] == 0.0) {
                    throw std::runtime_error("Zero diagonal in Cholesky decomposition");
                }
                L[i][j] = (cov_matrix[i][j] - s) / L[j][j];
            }
        }
    }
    return L;
}

double annualized_return(const std::vector<double>& daily_returns) {
    if (daily_returns.empty()) return 0.0;
    double mean = std::accumulate(daily_returns.begin(), daily_returns.end(), 0.0)
                  / daily_returns.size();
    return mean * 252.0;
}

double annualized_volatility(const std::vector<double>& daily_returns) {
    if (daily_returns.size() < 2) return 0.0;
    double mean = std::accumulate(daily_returns.begin(), daily_returns.end(), 0.0)
                  / daily_returns.size();
    double variance = 0.0;
    for (double r : daily_returns) {
        double diff = r - mean;
        variance += diff * diff;
    }
    variance /= (daily_returns.size() - 1);
    return std::sqrt(variance * 252.0);
}

} // namespace risklab
