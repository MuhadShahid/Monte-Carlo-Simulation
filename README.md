# RiskLab — Monte Carlo Portfolio Simulator

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.16%2B-064F8C?style=for-the-badge&logo=cmake&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)
![Build](https://img.shields.io/badge/Build-Passing-22C55E?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-37%20Passing-22C55E?style=for-the-badge)

**A quantitative finance tool combining a high-performance C++17 simulation engine with an interactive Streamlit frontend.**

*Developed by Muhad Shahid*

</div>

---

## Overview

RiskLab fetches live historical equity data, runs configurable Monte Carlo simulations to project multi-asset portfolio outcomes using Geometric Brownian Motion with correlated asset returns, computes the efficient frontier via Modern Portfolio Theory, calculates Value at Risk and Conditional VaR, and presents everything through interactive Plotly visualizations — all without an API key.

The project is structured as a portfolio and interview piece for quantitative finance and investment banking roles. Every architectural decision exists for a specific reason: C++ for the simulation inner loop, pybind11 for a zero-overhead Python bridge, and Streamlit for rapid interactive visualization without JavaScript complexity.

---

## Architecture

```
risklab/
├── CMakeLists.txt                  # Top-level CMake configuration
├── setup.py                        # Alternative pip install for the C++ extension
├── requirements.txt                # Python dependencies
├── README.md
│
├── engine/                         # C++17 simulation engine (no external deps)
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── random_engine.h         # Mersenne Twister RNG, Box-Muller sampling
│   │   ├── statistics.h            # Covariance matrix, Cholesky decomposition
│   │   ├── portfolio.h             # Asset and Portfolio classes
│   │   ├── monte_carlo.h           # GBM simulator, SimulationResult struct
│   │   └── risk_metrics.h          # VaR, CVaR, max drawdown, Sharpe
│   ├── src/
│   │   ├── random_engine.cpp
│   │   ├── statistics.cpp
│   │   ├── portfolio.cpp
│   │   ├── monte_carlo.cpp
│   │   └── risk_metrics.cpp
│   └── bindings/
│       └── py_risklab.cpp          # pybind11 module: risklab_engine
│
├── app/                            # Python / Streamlit frontend
│   ├── main.py                     # Streamlit entry point
│   ├── data_loader.py              # yfinance data fetching and preprocessing
│   ├── optimizer.py                # Efficient frontier via random portfolio sampling
│   ├── simulator.py                # Orchestrates C++ simulation runs
│   └── charts.py                   # Plotly visualization functions
│
└── tests/
    ├── test_engine.py              # 30 tests against the C++ module
    └── test_data_loader.py         # 7 tests for the data pipeline
```

---

## Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| Simulation Engine | C++17 | 10,000 paths x 252 days x 5 assets runs in under 600ms; industry standard in quant finance |
| Python Binding | pybind11 | Direct in-process function calls — no serialization, no IPC, zero overhead |
| Build System | CMake 3.16+ / setup.py | Standard C++ build tooling; handles pybind11 integration cleanly |
| Frontend | Streamlit | Interactive data app with no JavaScript required |
| Visualization | Plotly | Interactive charts: fan charts, 3D scatter, histograms with hover |
| Data Ingestion | yfinance | Free, keyless Yahoo Finance wrapper |
| Numerics (Python) | NumPy, Pandas | Data preprocessing and orchestration only — no simulation logic |

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9 or higher |
| C++ compiler | g++ 9+ or clang 10+ with C++17 support |
| CMake | 3.16+ (optional — `setup.py` builds without it) |
| pip | Any recent version |

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/muhadshahid/risklab.git
cd risklab
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Build the C++ simulation engine

**Option A — CMake (recommended for development):**

```bash
mkdir build && cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python3)
make -j$(nproc)
# Compiled .so is written to app/ automatically
```

**Option B — pip install (simpler):**

```bash
pip install -e .
```

### 4. Launch the application

```bash
streamlit run app/main.py
```

The app will be available at `http://localhost:8501`.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Expected output: **37 tests passing** across two test modules.

```
tests/test_engine.py       30 passed
tests/test_data_loader.py   7 passed
```

---

## Usage

### Sidebar Controls

| Control | Description |
|---|---|
| Tickers | Comma-separated ticker symbols (e.g. `AAPL, MSFT, GOOGL`) |
| Historical period | Data window for computing returns and covariance (1y to 10y) |
| Number of simulations | Monte Carlo paths: 1,000 to 50,000 |
| Time horizon | Projection horizon in months (6 to 60) |
| Initial portfolio value | Starting capital in USD |
| Risk-free rate | Annual risk-free rate for Sharpe calculation |
| Weight allocation | Manual sliders, equal weight, or max-Sharpe optimal |
| Fixed seed | Toggle reproducibility of simulation results |

### Output Sections

**Section 1 — Historical Context**
Normalized cumulative return chart, asset correlation heatmap, and annualized return/volatility table.

**Section 2 — Efficient Frontier**
Scatter plot of 5,000 random portfolios colored by Sharpe ratio. Maximum Sharpe and minimum volatility portfolios are annotated with their weights.

**Section 3 — Monte Carlo Simulation**
Fan chart of simulated paths with 5th/25th/75th/95th percentile bands and bold median path. Terminal value distribution histogram with VaR and CVaR markers. Maximum drawdown distribution histogram.

**Section 4 — Risk Summary**
Metric cards for: annualized expected return, annualized volatility, Sharpe ratio, probability of loss, VaR 95%, VaR 99%, CVaR 95%, and median terminal value.

---

## Financial Theory

### Geometric Brownian Motion

Asset prices are evolved using the log-normal GBM model:

```
S_{t+1} = S_t * exp( (mu - 0.5 * sigma^2) * dt  +  L*z )
```

where `mu` is the annualized expected return, `sigma` is annualized volatility, `dt = 1/252`, `L` is the lower Cholesky factor of the daily covariance matrix, and `z ~ N(0, I)`.

The `mu - 0.5*sigma^2` term is the Ito correction ensuring that the *expected* portfolio value grows at rate `mu`, not the median. This is the standard model underlying Black-Scholes options pricing.

### Correlated Asset Returns

Assets are not independent. The covariance matrix `Sigma` is estimated from historical daily log returns. Cholesky decomposition `Sigma = L * L^T` transforms independent standard normal draws into correlated draws with the correct covariance structure. This is the standard technique in multi-asset Monte Carlo simulation and risk models.

### Efficient Frontier and Modern Portfolio Theory

Markowitz (1952) showed that for any target expected return there exists a portfolio that minimizes risk. The set of such portfolios forms the efficient frontier. RiskLab approximates this by sampling 5,000 random weight combinations and computing expected return, volatility, and Sharpe ratio for each using the C++ Portfolio class. Two portfolios are highlighted: maximum Sharpe ratio and minimum volatility.

### Value at Risk (VaR) and Conditional VaR (CVaR)

**VaR at confidence level `c`**: The loss that is not exceeded with probability `c`. For 95% VaR, sort all simulated terminal values in ascending order and take the 5th percentile. If the 5th percentile terminal value is \$92,000 on a \$100,000 portfolio, VaR = \$8,000.

**CVaR (Expected Shortfall)**: The mean loss across all scenarios worse than the VaR threshold. CVaR is always greater than or equal to VaR at the same confidence level and is considered a more coherent risk measure because it captures the severity of tail losses, not merely their threshold.

---

## Implementation Notes

### Why C++ for the Simulation Loop

10,000 simulations x 252 trading days x 5 assets means 12.6 million GBM steps per run. In C++ with -O3, this completes in under 600ms including Python data transfer overhead. An equivalent pure Python or NumPy implementation would be approximately 10-50x slower and would noticeably lag on user interaction.

### No External C++ Dependencies

All matrix operations (covariance, Cholesky decomposition) use `std::vector<std::vector<double>>` and the C++ standard library only. For 5x5 to 10x10 matrices, Eigen or BLAS would add build complexity with no measurable performance benefit.

### pybind11 Integration

The compiled extension (`risklab_engine.cpython-*.so`) is a direct in-process Python module. Calling `MonteCarloSimulator.run()` from Python is equivalent to a function call — there is no serialization, no subprocess, no network. This is the correct architecture for performance-critical Python/C++ integration.

---

## Project Structure Rationale

```
Simulation logic    →  C++ only (engine/)
Data orchestration  →  Python only (app/simulator.py, app/optimizer.py)
Visualization       →  Python only (app/charts.py)
User interface      →  Streamlit only (app/main.py)
```

Each layer has a single responsibility. The C++ engine exposes a clean typed API via pybind11. Python orchestrates data flow. Streamlit handles rendering. No layer bleeds into another.

---

## Author

**Muhad Shahid**

---

## License

This project is licensed under the MIT License.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

> This tool is for educational and research purposes only. It does not constitute financial advice.
