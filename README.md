# Continuous Multi-Objective Bayesian Optimizer (Gryffin Architecture)

A general-purpose, high-performance Bayesian Optimization library designed for autonomous experiment planning, parameter optimization, and multi-objective decision making in physical science and engineering experiments.

Derived from the principles of **Gryffin** and **Phoenics**, this package is specifically optimized for **continuous continuous numerical features** (e.g. Temperature, pH, Pressure, Concentrations, Flow Rate, Time, Agitation Speed) and multi-objective targets.

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

## Quickstart Example

```python
from bayesian_optimizer import BayesianOptimizer

# 1. Configure 8 continuous experimental parameters
parameters = [
    {"name": "temperature", "type": "continuous", "low": 20.0, "high": 120.0},
    {"name": "ph",          "type": "continuous", "low": 2.0,  "high": 12.0},
    {"name": "pressure",    "type": "continuous", "low": 1.0,  "high": 30.0},
    {"name": "conc1",       "type": "continuous", "low": 0.1,  "high": 5.0},
    {"name": "conc2",       "type": "continuous", "low": 0.05, "high": 2.0},
    {"name": "flow_rate",   "type": "continuous", "low": 0.5,  "high": 10.0},
    {"name": "time",        "type": "continuous", "low": 10.0, "high": 180.0},
    {"name": "agit_speed",  "type": "continuous", "low": 100.0,"high": 1200.0},
]

# 2. Configure 5 target objectives
objectives = [
    {"name": "yield",     "goal": "max", "tolerance": 0.05},
    {"name": "purity",    "goal": "max", "tolerance": 0.02},
    {"name": "cost",      "goal": "min", "tolerance": 0.10},
    {"name": "byproduct", "goal": "min", "tolerance": 0.05},
    {"name": "energy",    "goal": "min", "tolerance": 0.10},
]

config = {
    "general": {
        "sampling_strategies": 2, # 1 Exploitation (λ > 0), 1 Exploration (λ < 0)
        "batches": 2,             # 2 batch recommendations per loop
        "feas_approach": "fwa",   # Feasibility-weighted acquisition
        "obj_transform": "sqrt",  # Objective transform
    },
    "model": {
        "num_epochs": 800,
        "learning_rate": 0.05,
        "num_draws": 300,
        "hidden_shape": 12,
        "num_layers": 3
    },
    "parameters": parameters,
    "objectives": objectives
}

# 3. Optional user constraint function
def domain_constraints(p):
    # Enforce temperature * pressure <= 1800 and pH >= 3.5
    if p['temperature'] * p['pressure'] > 1800.0:
        return False
    if p['ph'] < 3.5:
        return False
    return True

# 4. Instantiate Optimizer
optimizer = BayesianOptimizer(config_dict=config, known_constraints=domain_constraints)

# 5. Provide historical observation dictionary list
observations = [
    {
        "temperature": 60.0, "ph": 7.0, "pressure": 10.0, "conc1": 1.5,
        "conc2": 0.4, "flow_rate": 3.0, "time": 45.0, "agit_speed": 400.0,
        "yield": 88.5, "purity": 96.2, "cost": 14.5, "byproduct": 2.1, "energy": 8.4
    }
]

# 6. Request recommended experimental feature values for the next trial
recommendations = optimizer.recommend(observations=observations)
print("Recommended Trial Features:", recommendations)
```

---

## Project Structure

```
bayesian_optimizer/
├── __init__.py                                 # Package export
├── optimizer.py                                # Main high-level user interface
├── config.py                                   # Parameter & objective configuration parser
├── observation_processor.py                    # Multi-objective Chimera scalarizer & boundary mirroring
├── random_sampler.py                           # Random/perturbed candidate generator with constraint checks
├── sample_selector.py                          # Diversity-penalized recommendation selection
├── run_example.py                              # Executable end-to-end multi-objective test script
├── ARCHITECTURE_AND_PREDICTION_REPORT.md      # Full architecture & prediction report
├── DETAILED_ARCHITECTURE_AND_PREDICTION_GUIDE.md # Exhaustive mathematical & algorithmic guide
├── bayesian_network/                           # Bayesian Neural Network & Kernel Density module
│   ├── bnn_trainer.py                          # PyTorch Variational Bayes training loop
│   ├── numpy_graph.py                          # Posterior draw computation graph
│   ├── kernel_evaluator.py                     # Vectorized NumPy Nadaraya-Watson KDE
│   └── bayesian_network.py                     # Bayesian network coordinator
└── acquisition/                                # Acquisition function optimization module
    ├── adam_optimizer.py                       # Finite-difference Adam gradient optimizer
    └── acquisition.py                          # λ-parameter acquisition function
```

---

## Documentation

For a comprehensive mathematical breakdown, theoretical foundations, derivation of kernel regression, acquisition functions, and step-by-step prediction flow, consult:
- **[DETAILED_ARCHITECTURE_AND_PREDICTION_GUIDE.md](bayesian_optimizer/DETAILED_ARCHITECTURE_AND_PREDICTION_GUIDE.md)**
- **[ARCHITECTURE_AND_PREDICTION_REPORT.md](bayesian_optimizer/ARCHITECTURE_AND_PREDICTION_REPORT.md)**

---

## Citation & License

This project implements the continuous optimization logic from **Gryffin**:
- *Hase, F., Aldeghi, M., et al. "Gryffin: An algorithm for Bayesian optimization of continuous and categorical variables." Applied Physics Reviews (2021).*

Licensed under the Apache License 2.0.
