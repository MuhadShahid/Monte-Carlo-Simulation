#pragma once
#include "portfolio.h"
#include "random_engine.h"
#include <vector>

namespace risklab {

struct SimulationResult {
    // paths[i] is the portfolio value path for simulation i (length = horizon_days + 1)
    // paths[i][0] = initial_value, paths[i][T] = terminal value
    std::vector<std::vector<double>> paths;

    // Convenience: terminal value of each path
    std::vector<double> terminal_values;
};

class MonteCarloSimulator {
public:
    MonteCarloSimulator(
        int num_simulations,
        int horizon_days,
        const Portfolio& portfolio,
        RandomEngine& rng,
        double initial_value = 1.0
    );

    // Run the simulation; returns paths and terminal values
    // Uses GBM: S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    // where Z are correlated draws from the Cholesky factor of the cov matrix
    SimulationResult run();

private:
    int num_simulations_;
    int horizon_days_;
    const Portfolio& portfolio_;
    RandomEngine& rng_;
    double initial_value_;
};

} // namespace risklab
