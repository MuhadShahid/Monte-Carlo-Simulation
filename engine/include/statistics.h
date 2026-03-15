#pragma once
#include <vector>
#include <string>

namespace risklab {

// Convert a price series to log returns: log(P_t / P_{t-1})
std::vector<double> compute_log_returns(const std::vector<double>& prices);

// Compute sample covariance matrix from a matrix of return series
// returns[i] is the full return series for asset i
// Returns an n x n symmetric covariance matrix
std::vector<std::vector<double>> compute_covariance_matrix(
    const std::vector<std::vector<double>>& returns);

// Cholesky decomposition: returns lower triangular L such that A = L * L^T
// Throws std::runtime_error if matrix is not positive definite
std::vector<std::vector<double>> cholesky_decompose(
    const std::vector<std::vector<double>>& cov_matrix);

// Compute annualized mean return (multiply daily mean by 252)
double annualized_return(const std::vector<double>& daily_returns);

// Compute annualized volatility (multiply daily std dev by sqrt(252))
double annualized_volatility(const std::vector<double>& daily_returns);

} // namespace risklab
