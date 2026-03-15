set(ENGINE_SOURCES
    src/random_engine.cpp
    src/statistics.cpp
    src/portfolio.cpp
    src/monte_carlo.cpp
    src/risk_metrics.cpp
)

set(BINDING_SOURCES
    bindings/py_risklab.cpp
)

# Build the pybind11 extension module
pybind11_add_module(risklab_engine
    ${ENGINE_SOURCES}
    ${BINDING_SOURCES}
)

target_include_directories(risklab_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(risklab_engine PRIVATE
    -Wall -Wextra -Wpedantic
)

# Place the compiled .so in the app/ directory for direct import
set_target_properties(risklab_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/app
)
