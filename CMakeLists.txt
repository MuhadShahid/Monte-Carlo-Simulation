cmake_minimum_required(VERSION 3.16)
project(risklab VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Release build by default for performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Try to find pybind11 via cmake first, then fall back to python -m pybind11
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_NOT_FOUND
    )
    if(PYBIND11_NOT_FOUND)
        message(FATAL_ERROR "pybind11 not found. Install it with: pip install pybind11")
    endif()
    find_package(pybind11 CONFIG REQUIRED PATHS ${PYBIND11_CMAKE_DIR})
endif()

add_subdirectory(engine)
