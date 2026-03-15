#pragma once
#include <string>
#include <vector>

namespace risklab {

struct Asset {
    std::string ticker;
    double annualized_return;  // annualized expected return (e.g., 0.10 = 10%)
    double annualized_volatility; // annualized volatility (e.g., 0.20 = 20%)
};

class Portfolio {
public:
    Portfolio() = default;

    void add_asset(const Asset& asset);
    void set_weights(const std::vector<double>& weights);
    void set_covariance_matrix(const std::vector<std::vector<double>>& cov);

    const std::vector<Asset>& assets() const { return assets_; }
    const std::vector<double>& weights() const { return weights_; }
    const std::vector<std::vector<double>>& covariance_matrix() const { return cov_matrix_; }
    std::size_t num_assets() const { return assets_.size(); }

    // Weighted portfolio expected return
    double expected_return() const;

    // Portfolio volatility: sqrt(w^T * Sigma * w)
    // Covariance matrix must be set first
    double volatility() const;

    // Sharpe ratio: (E[r] - rf) / sigma
    double sharpe_ratio(double risk_free_rate) const;

private:
    std::vector<Asset> assets_;
    std::vector<double> weights_;
    std::vector<std::vector<double>> cov_matrix_;
};

} // namespace risklab
