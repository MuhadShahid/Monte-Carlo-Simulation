#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "random_engine.h"
#include "statistics.h"
#include "portfolio.h"
#include "monte_carlo.h"
#include "risk_metrics.h"

namespace py = pybind11;

PYBIND11_MODULE(risklab_engine, m) {
    m.doc() = "RiskLab C++ Monte Carlo simulation engine";

    // ── RandomEngine ─────────────────────────────────────────────────────────
    py::class_<risklab::RandomEngine>(m, "RandomEngine",
        "Mersenne Twister RNG with Box-Muller normal sampling")
        .def(py::init<uint64_t>(), py::arg("seed") = 42,
             "Construct with optional seed for reproducibility")
        .def("reseed", &risklab::RandomEngine::reseed, py::arg("seed"),
             "Re-seed the RNG")
        .def("normal", &risklab::RandomEngine::normal,
             "Draw a single standard normal variate")
        .def("normal_vector", &risklab::RandomEngine::normal_vector, py::arg("n"),
             "Draw n independent standard normal variates")
        .def("correlated_normals", &risklab::RandomEngine::correlated_normals,
             py::arg("cholesky_L"),
             "Generate correlated normal variates from lower Cholesky factor L");

    // ── Statistics functions ──────────────────────────────────────────────────
    m.def("compute_log_returns", &risklab::compute_log_returns, py::arg("prices"),
          "Convert price series to log returns: log(P_t / P_{t-1})");

    m.def("compute_covariance_matrix", &risklab::compute_covariance_matrix,
          py::arg("returns"),
          "Compute sample covariance matrix from list of return series (one per asset)");

    m.def("cholesky_decompose", &risklab::cholesky_decompose,
          py::arg("cov_matrix"),
          "Lower triangular Cholesky decomposition of a positive definite matrix");

    m.def("annualized_return", &risklab::annualized_return, py::arg("daily_returns"),
          "Annualized mean return from daily log returns (multiply by 252)");

    m.def("annualized_volatility", &risklab::annualized_volatility, py::arg("daily_returns"),
          "Annualized volatility from daily log returns (multiply std dev by sqrt(252))");

    // ── Asset ─────────────────────────────────────────────────────────────────
    py::class_<risklab::Asset>(m, "Asset", "Represents a single investable asset")
        .def(py::init<>())
        .def_readwrite("ticker", &risklab::Asset::ticker)
        .def_readwrite("annualized_return", &risklab::Asset::annualized_return)
        .def_readwrite("annualized_volatility", &risklab::Asset::annualized_volatility)
        .def("__repr__", [](const risklab::Asset& a) {
            return "<Asset ticker='" + a.ticker
                + "' return=" + std::to_string(a.annualized_return)
                + " vol=" + std::to_string(a.annualized_volatility) + ">";
        });

    // ── Portfolio ─────────────────────────────────────────────────────────────
    py::class_<risklab::Portfolio>(m, "Portfolio",
        "Multi-asset portfolio with weights and covariance matrix")
        .def(py::init<>())
        .def("add_asset", &risklab::Portfolio::add_asset, py::arg("asset"),
             "Add an Asset to the portfolio")
        .def("set_weights", &risklab::Portfolio::set_weights, py::arg("weights"),
             "Set portfolio weights (must sum to 1.0 and match number of assets)")
        .def("set_covariance_matrix", &risklab::Portfolio::set_covariance_matrix,
             py::arg("cov_matrix"),
             "Set the daily covariance matrix (n x n, one row/col per asset)")
        .def("assets", &risklab::Portfolio::assets,
             "Return list of Asset objects")
        .def("weights", &risklab::Portfolio::weights,
             "Return weight vector")
        .def("covariance_matrix", &risklab::Portfolio::covariance_matrix,
             "Return the covariance matrix")
        .def("num_assets", &risklab::Portfolio::num_assets,
             "Number of assets in the portfolio")
        .def("expected_return", &risklab::Portfolio::expected_return,
             "Annualized portfolio expected return (weighted sum)")
        .def("volatility", &risklab::Portfolio::volatility,
             "Annualized portfolio volatility (sqrt of w^T * Sigma * w, annualized)")
        .def("sharpe_ratio", &risklab::Portfolio::sharpe_ratio, py::arg("risk_free_rate"),
             "Sharpe ratio: (E[r] - rf) / sigma");

    // ── SimulationResult ──────────────────────────────────────────────────────
    py::class_<risklab::SimulationResult>(m, "SimulationResult",
        "Output of a Monte Carlo simulation run")
        .def_readwrite("paths", &risklab::SimulationResult::paths,
                       "List of lists: paths[i] is the portfolio value path for simulation i")
        .def_readwrite("terminal_values", &risklab::SimulationResult::terminal_values,
                       "Terminal portfolio value for each simulation");

    // ── MonteCarloSimulator ───────────────────────────────────────────────────
    py::class_<risklab::MonteCarloSimulator>(m, "MonteCarloSimulator",
        "GBM-based Monte Carlo simulator for a multi-asset portfolio")
        .def(py::init<int, int, const risklab::Portfolio&, risklab::RandomEngine&, double>(),
             py::arg("num_simulations"),
             py::arg("horizon_days"),
             py::arg("portfolio"),
             py::arg("rng"),
             py::arg("initial_value") = 1.0,
             "Construct the simulator. initial_value defaults to 1.0 (normalized).")
        .def("run", &risklab::MonteCarloSimulator::run,
             "Run the simulation and return SimulationResult");

    // ── Risk metrics ──────────────────────────────────────────────────────────
    m.def("value_at_risk", &risklab::value_at_risk,
          py::arg("terminal_values"), py::arg("confidence"), py::arg("initial_value") = 1.0,
          "Value at Risk: loss at given confidence level (positive = loss)");

    m.def("conditional_var", &risklab::conditional_var,
          py::arg("terminal_values"), py::arg("confidence"), py::arg("initial_value") = 1.0,
          "Conditional VaR (Expected Shortfall): mean loss beyond VaR threshold");

    m.def("max_drawdown", &risklab::max_drawdown, py::arg("path"),
          "Maximum peak-to-trough drawdown fraction in a single simulation path");

    m.def("sharpe_from_paths", &risklab::sharpe_from_paths,
          py::arg("paths"), py::arg("risk_free_rate") = 0.05,
          "Annualized Sharpe ratio computed from simulated portfolio paths");
}
