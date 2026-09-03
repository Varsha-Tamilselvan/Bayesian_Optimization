# Continuous Multi-Objective Bayesian Optimizer (Gryffin Architecture)

A general-purpose, high-performance Bayesian Optimization library designed for autonomous experiment planning, parameter optimization, and multi-objective decision making in physical science and engineering experiments.

Derived from the principles of **Gryffin** and **Phoenics**, this package is specifically optimized for **continuous numerical features** (e.g. Temperature, pH, Pressure, Concentrations, Flow Rate, Time, Agitation Speed) and multi-objective targets.

---

## Key Features

- **Linear-Scaling Bayesian Neural Network (BNN)**: Replaces standard Gaussian Process (GP) regression to achieve $\mathcal{O}(N)$ scaling instead of $\mathcal{O}(N^3)$ computational cost.
- **Hierarchical Multi-Objective Scalarization**: Integrates Chimera-style multi-objective scalarization to handle up to 5+ target columns with user-defined tolerances and optimization goals (`min` / `max`).
- **Custom Physical Constraint Enforcement**: Evaluates user-defined analytical constraints $g(\mathbf{x}) \le 0$ during random sampling, perturbation, and acquisition optimization.
- **Feasibility Classification**: Handles infeasible experimental conditions (e.g. experiments resulting in `NaN` or process failures) using Bayesian Kernel Classification (`fwa`, `fia`, `fca`).
- **Diversity-Aware Sample Selection**: Recommends batch experiments that balance performance optimization with spatial coverage, preventing clustered recommendations.
- **Pure Python/PyTorch/NumPy Engine**: No Cython, C compilation, or external C-extensions required. Fully portable across Windows, Linux, and macOS.

---

## Installation & Requirements

### Requirements
- Python >= 3.8
- `numpy >= 1.20.0`
- `torch >= 1.9.0`
- `scipy >= 1.7.0`

### Quick Setup

```bash
git clone https://github.com/Varsha-Tamilselvan/bayesian-optimization-model.git
cd bayesian-optimization-model
pip install -r bayesian_optimizer/requirements.txt
```

---

## Documentation

For a comprehensive mathematical breakdown, theoretical foundations, derivation of kernel regression, acquisition functions, and step-by-step prediction flow, consult:
- **[DETAILED_ARCHITECTURE_AND_PREDICTION_GUIDE.md](DETAILED_ARCHITECTURE_AND_PREDICTION_GUIDE.md)**
- **[ARCHITECTURE_AND_PREDICTION_REPORT.md](ARCHITECTURE_AND_PREDICTION_REPORT.md)**
