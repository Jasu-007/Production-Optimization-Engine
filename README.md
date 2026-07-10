# Enterprise Operations & Linear Solver Engine (C++)

A custom-built, zero-dependency linear optimization engine designed to autonomously resolve multi-variable enterprise operations matrices and supply chain logistics problems.

## Architecture & Algorithms
* **Memory Architecture:** Implements a highly optimized 1D flat-matrix memory layout to maximize CPU cache locality and ensure O(1) spatial access.
* **Core Solvers:**
  * **Vogel’s Approximation Method (VAM):** Generates near-optimal Initial Basic Feasible Solutions (IBFS) for distribution networks.
  * **Modified Distribution Method (MODI / UV):** Iteratively evaluates opportunity costs to drive the IBFS to the absolute global mathematical optimum.
  * **Big-M Method (Simplex):** Handles mixed-constraint linear programming (<=, >=, ==) through artificial variable perturbation.
* **Edge-Case Handling:** Engineered to automatically resolve matrix degeneracy via epsilon perturbation, unboundedness, and unbalanced supply/demand constraints without crashing.

## Performance Benchmarks
Compiled via `g++ -O3`. Tested on dynamically generated, perfectly balanced supply-demand transportation matrices.
* **100x100 Matrix** (10,000 variables): `~59 ms`
* **500x500 Matrix** (250,000 variables): `~6.9 seconds`
* **1000x1000 Matrix** (1,000,000 variables): `~68.9 seconds`

## Build & Execute
```bash
# Compile with maximum optimizations
g++ main.cpp BigMSolver.cpp VogelSolver.cpp -std=c++20 -O3 -Wall -Wextra -Wpedantic -o engine

# Execute the mathematical validation suite
./engine