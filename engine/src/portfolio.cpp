#include "portfolio.h"
#include <stdexcept>
#include <cmath>
#include <numeric>

namespace risklab {

void Portfolio::add_asset(const Asset& asset) {
    assets_.push_back(asset);
}

void Portfolio::set_weights(const std::vector<double>& weights) {
    if (weights.size() != assets_.size()) {
        throw std::invalid_argument("Weight vector size must match number of assets");
    }
    weights_ = weights;
}

void Portfolio::set_covariance_matrix(const std::vector<std::vector<double>>& cov) {
    if (cov.size() != assets_.size()) {
        throw std::invalid_argument("Covariance matrix size must match number of assets");
    }
    cov_matrix_ = cov;
}

double Portfolio::expected_return() const {
    if (weights_.size() != assets_.size()) {
        throw std::runtime_error("Weights not set");
    }
    double result = 0.0;
    for (std::size_t i = 0; i < assets_.size(); ++i) {
        result += weights_[i] * assets_[i].annualized_return;
    }
    return result;
}

double Portfolio::volatility() const {
    if (weights_.size() != assets_.size()) {
        throw std::runtime_error("Weights not set");
    }
    if (cov_matrix_.empty()) {
        throw std::runtime_error("Covariance matrix not set");
    }
    std::size_t n = assets_.size();
    // variance = w^T * Sigma * w
    double variance = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            variance += weights_[i] * weights_[j] * cov_matrix_[i][j];
        }
    }
    // cov_matrix_ is daily covariance; annualize
    return std::sqrt(variance * 252.0);
}

double Portfolio::sharpe_ratio(double risk_free_rate) const {
    double sigma = volatility();
    if (sigma == 0.0) return 0.0;
    return (expected_return() - risk_free_rate) / sigma;
}

} // namespace risklab
