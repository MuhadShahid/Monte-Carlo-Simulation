"""Alternative build via pip install . or pip install -e ."""
from glob import glob
from setuptools import setup

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:
    raise RuntimeError("pybind11 is required: pip install pybind11")

ext_modules = [
    Pybind11Extension(
        "risklab_engine",
        sorted(glob("engine/src/*.cpp") + glob("engine/bindings/*.cpp")),
        include_dirs=["engine/include"],
        cxx_std=17,
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="risklab_engine",
    version="1.0.0",
    description="RiskLab C++ Monte Carlo simulation engine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.9",
)
