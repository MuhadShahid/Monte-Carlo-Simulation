#pragma once
#include <random>
#include <vector>
#include <cstdint>

namespace risklab {

class RandomEngine {
public:
    explicit RandomEngine(uint64_t seed = 42);

    // Generate a single standard normal variate using Box-Muller transform
    double normal();

    // Generate n independent standard normal variates
    std::vector<double> normal_vector(std::size_t n);

    // Generate correlated normal variates given the lower Cholesky factor L
    // Returns L * z where z ~ N(0, I)
    std::vector<double> correlated_normals(const std::vector<std::vector<double>>& L);

    void reseed(uint64_t seed);

private:
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uniform_{0.0, 1.0};
    bool has_spare_ = false;
    double spare_ = 0.0;
};

} // namespace risklab
