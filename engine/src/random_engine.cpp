#include "random_engine.h"
#include <cmath>
#include <stdexcept>

namespace risklab {

RandomEngine::RandomEngine(uint64_t seed) : rng_(seed) {}

void RandomEngine::reseed(uint64_t seed) {
    rng_.seed(seed);
    has_spare_ = false;
}

double RandomEngine::normal() {
    // Box-Muller transform with spare value caching
    if (has_spare_) {
        has_spare_ = false;
        return spare_;
    }

    double u, v, s;
    do {
        u = uniform_(rng_) * 2.0 - 1.0;
        v = uniform_(rng_) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    double mul = std::sqrt(-2.0 * std::log(s) / s);
    spare_ = v * mul;
    has_spare_ = true;
    return u * mul;
}

std::vector<double> RandomEngine::normal_vector(std::size_t n) {
    std::vector<double> result(n);
    for (std::size_t i = 0; i < n; ++i) {
        result[i] = normal();
    }
    return result;
}

std::vector<double> RandomEngine::correlated_normals(
    const std::vector<std::vector<double>>& L)
{
    std::size_t n = L.size();
    if (n == 0) throw std::invalid_argument("Cholesky factor is empty");

    // Generate independent standard normal vector z
    std::vector<double> z = normal_vector(n);

    // Compute L * z (lower triangular matrix-vector multiply)
    std::vector<double> result(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            result[i] += L[i][j] * z[j];
        }
    }
    return result;
}

} // namespace risklab
