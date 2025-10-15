# Monte-Carlo-Integrator
Monte Carlo integration for high-dimensional functions

Features:
 - Monte Carlo integration in arbitrary dimensions.
 - Core computation in C++ for speed.
 - OpenMP multi-threading to utilize multiple CPU cores.
 - Batching in Python to reduce memory usage for large sample sizes.
 - Easy Python interface for flexible integration.

Setup & Compilation:

1. Install pybind
   pip install pybind11
2. Compile the C++ Module
   c++ -O3 -Wall -shared -std=c++17 -fopenmp -fPIC \
   $(python3 -m pybind11 --includes) monte_carlo_batch.cpp -o monte_carlo_batch$(python3-config --extension-suffix)
