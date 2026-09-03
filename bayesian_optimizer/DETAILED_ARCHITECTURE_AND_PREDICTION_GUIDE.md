# Comprehensive Theoretical & Algorithmic Guide
## Continuous Multi-Objective Bayesian Optimization Model (Gryffin-Based Framework)

---

## Table of Contents
1. [Introduction & Problem Statement](#1-introduction--problem-statement)
2. [Why Linear-Scaling BNNs Over Gaussian Processes](#2-why-linear-scaling-bnns-over-gaussian-processes)
3. [Deep-Dive System Architecture](#3-deep-dive-system-architecture)
4. [Mathematical Formalism & Derivations](#4-mathematical-formalism--derivations)
   - 4.1 [Observation Processing & Boundary Reflection](#41-observation-processing--boundary-reflection)
   - 4.2 [Hierarchical Chimera Multi-Objective Scalarization](#42-hierarchical-chimera-multi-objective-scalarization)
   - 4.3 [Bayesian Neural Network (BNN) & Variational Inference](#43-bayesian-neural-network-bnn--variational-inference)
   - 4.4 [Posterior Draw Graph & Kernel Parameter Extraction](#44-posterior-draw-graph--kernel-parameter-extraction)
   - 4.5 [Nadaraya-Watson Kernel Density Estimator (KDE)](#45-nadaraya-watson-kernel-density-estimator-kde)
   - 4.6 [Bayesian Kernel Classification for Infeasible Regions](#46-bayesian-kernel-classification-for-infeasible-regions)
   - 4.7 [Gryffin Acquisition Function & λ Parameter Dynamics](#47-gryffin-acquisition-function--λ-parameter-dynamics)
   - 4.8 [Finite-Difference Adam Optimizer with Constraints](#48-finite-difference-adam-optimizer-with-constraints)
   - 4.9 [Distance-Penalized Diversity Sample Selector](#49-distance-penalized-diversity-sample-selector)
5. [Step-by-Step Numerical Prediction & Optimization Walkthrough](#5-step-by-step-numerical-prediction--optimization-walkthrough)
6. [Code Component Walkthrough](#6-code-component-walkthrough)
7. [Constraint Capability & Handling Infeasible Data](#7-constraint-capability--handling-infeasible-data)
8. [Troubleshooting & Performance Tuning Guide](#8-troubleshooting--performance-tuning-guide)

---

## 1. Introduction & Problem Statement

In physical chemical synthesis, materials science, and industrial bio-process engineering, finding optimal experimental feature settings (e.g. Temperature, pH, Pressure, Concentrations, Flow Rates, Time, Agitation Speed) is a complex challenge.

Specifically, the user's setup involves:
- **8 Continuous Features** $\mathbf{x} = [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8]^\top \in \Omega \subset \mathbb{R}^8$.
- **5 Target Objectives** $\mathbf{y} = [y_1, y_2, y_3, y_4, y_5]^\top$.
- **Known Physical Constraints**: Non-linear operational boundaries $g(\mathbf{x}) \le 0$ (e.g. $T \cdot P \le 1800$).
- **Infeasible Experimental Runs**: Conditions where experiments fail or output `NaN` values due to process instabilities.

This framework provides an autonomous experiment planning algorithm derived from **Gryffin** and **Phoenics** to systematically propose parameter sets that optimize all target columns simultaneously.

---

## 2. Why Linear-Scaling BNNs Over Gaussian Processes

Standard Bayesian Optimization relies on **Gaussian Process (GP)** surrogates. While GP surrogates work well for low-dimensional problems, they suffer from fundamental limitations:

| Metric / Aspect | Gaussian Process (GP) | Gryffin BNN Kernel Regression |
|---|---|---|
| **Computational Complexity** | $\mathcal{O}(N^3)$ matrix inversion cost | $\mathcal{O}(N)$ linear evaluation cost |
| **Memory Requirement** | $\mathcal{O}(N^2)$ kernel matrix storage | $\mathcal{O}(N \cdot D)$ linear storage |
| **Scalability with Samples** | Struggles beyond $N > 1000$ points | Easily scales to $N > 10,000+$ points |
| **Domain Boundary Handling** | Edge distortions without explicit tuning | Mirror-reflected continuous boundary kernels |
| **Feasibility Classification** | Separate independent GP Classifier | Integrated joint Bayesian kernel classifier |

By modeling the density over parameters using a Bayesian Neural Network, Gryffin retains kernel-based smoothness while remaining computationally light and linear-scaling.

---

## 3. Deep-Dive System Architecture

The following diagram illustrates the complete execution flow during a single optimization iteration:

```
[Raw Observations] -> [Boundary Mirroring] -> [Chimera Scalarizer]
                                                     |
                                                     v
                                          [Scalarized Target y & Feasibility s]
                                                     |
                                                     v
                                          [Bayesian Neural Network Training]
                                          (Variational Bayes ELBO Loss)
                                                     |
                                                     v
                                          [Posterior Graph Extraction]
                                          (Locations μ & Gamma Precisions τ)
                                                     |
                                                     v
                                          [Nadaraya-Watson Kernel Surrogate]
                                                     |
                                                     v
                                          [Acquisition Optimization (λ-Strategies)]
                                          (Adam Optimizer + Constraint Projection)
                                                     |
                                                     v
                                          [Distance-Penalized Sample Selector]
                                                     |
                                                     v
                                    [Final Recommended Feature Batches]
```

---

## 4. Mathematical Formalism & Derivations

### 4.1 Observation Processing & Boundary Reflection

To prevent boundary distortion at the domain limits $[x_{\text{low}, k}, x_{\text{high}, k}]$, observations located within 10% of domain edges are reflected across the boundary:

$$\tilde{x}_{\text{lower}, k} = x_{\text{low}, k} - (x_k - x_{\text{low}, k}) \quad \text{if } x_k < x_{\text{low}, k} + 0.1 \Delta x_k$$
$$\tilde{x}_{\text{upper}, k} = x_{\text{high}, k} + (x_{\text{high}, k} - x_k) \quad \text{if } x_k > x_{\text{high}, k} - 0.1 \Delta x_k$$

This ensures that the estimated kernel density has zero derivative normal to the domain boundary ($\frac{\partial p}{\partial x_k} = 0$), matching continuous physical boundary conditions.

---

### 4.2 Hierarchical Chimera Multi-Objective Scalarization

To optimize 5 objectives $\mathbf{y} = [y_1, y_2, y_3, y_4, y_5]^\top$, Chimera maps multi-objective vectors into a single scalar merit value $f(\mathbf{x}) \in [0, 1]$, where $0$ represents optimal performance and $1$ represents poor performance.

#### Step 1: Objective Normalization
For each target $i \in \{1, \dots, 5\}$:
$$z_i = \begin{cases} \frac{y_i - \min(y_i)}{\max(y_i) - \min(y_i)} & \text{if goal = min} \\[6pt] 1 - \frac{y_i - \min(y_i)}{\max(y_i) - \min(y_i)} & \text{if goal = max} \end{cases}$$

#### Step 2: Soft Hierarchical Folding
Objectives are prioritized in hierarchical order (from primary to secondary). Objectives are folded smoothly using tolerance $\tau_i$:
$$w_i = \frac{1}{2} \left[ 1 + \tanh\left( \frac{z_i - \tau_i}{\varepsilon} \right) \right]$$
$$m_i = w_i \cdot z_i + (1 - w_i) \cdot m_{i+1}$$

When $z_i \le \tau_i$ (objective $i$ satisfies tolerance), $w_i \approx 0$, allowing lower-priority objectives to drive optimization. When $z_i > \tau_i$, $w_i \approx 1$, forcing the optimizer to focus strictly on satisfying objective $i$.

---

### 4.3 Bayesian Neural Network (BNN) & Variational Inference

The BNN represents layer weights $w$ as Gaussian distributions $q_\theta(w) = \mathcal{N}(\mu_w, \sigma_w^2)$ with variational parameters $\theta = \{ \mu_w, \rho_w \}$, where $\sigma_w = \text{softplus}(\rho_w)$.

#### Training Objective (ELBO)
The parameters $\theta$ are trained using Adam by maximizing the Evidence Lower Bound (ELBO), or equivalently minimizing the loss:

$$\mathcal{L}(\theta) = -\sum_{i=1}^N \log p\left(y_i \mid \mathbf{x}_i, w\right) + \frac{1}{N} \text{KL}\left( q_\theta(w) \parallel p(w) \right)$$

where:
- $\log p\left(y_i \mid \mathbf{x}_i, w\right)$ is the Gaussian log-likelihood.
- $\text{KL}\left( q_\theta(w) \parallel p(w) \right)$ is the Kullback-Leibler divergence against standard normal prior $p(w) = \mathcal{N}(0, I)$:
$$\text{KL}\left( \mathcal{N}(\mu_w, \sigma_w^2) \parallel \mathcal{N}(0, 1) \right) = \frac{1}{2} \sum \left( \mu_w^2 + \sigma_w^2 - 1 - \log(\sigma_w^2) \right)$$

---

### 4.4 Posterior Draw Graph & Kernel Parameter Extraction

After $M$ training epochs, $S = 300\text{--}1000$ weight vectors $w^{(s)} \sim q_\theta(w)$ are drawn. For each draw $s \in \{1, \dots, S\}$ and observation $i \in \{1, \dots, N\}$, the network outputs:
1. **Kernel Locations**: $\mu_{s, i, k} = \Delta x_k \cdot \left( 1.2 \cdot \sigma(\text{BNN}(\mathbf{x}_i)) - 0.1 \right) + x_{\text{low}, k}$.
2. **Gamma Precisions**: $\tau_{s, i, k} \sim \text{Gamma}\left( 12 \left(\frac{N}{f_{\text{feas}}}\right)^2, \beta \right) / (\Delta x_k)^2$.

---

### 4.5 Nadaraya-Watson Kernel Density Estimator (KDE)

For any query point $\mathbf{x}^* \in \mathbb{R}^8$, the Gaussian kernel density $K(\mathbf{x}^*, \mathbf{x}_i)$ for observation $i$ is calculated by averaging across all BNN posterior draws:

$$K(\mathbf{x}^*, \mathbf{x}_i) = \frac{1}{S} \sum_{s=1}^S \prod_{k=1}^8 \left[ \frac{\sqrt{\tau_{s, i, k}}}{\sqrt{2\pi}} \exp\left( -\frac{1}{2} \tau_{s, i, k} \left( x_k^* - \mu_{s, i, k} \right)^2 \right) \right]$$

The predicted scalarized target value $\hat{y}(\mathbf{x}^*)$ is estimated via Nadaraya-Watson regression:

$$\hat{y}(\mathbf{x}^*) = \frac{\sum_{i \in \text{Feasible}} y_i \cdot K(\mathbf{x}^*, \mathbf{x}_i)}{\sum_{i \in \text{Feasible}} K(\mathbf{x}^*, \mathbf{x}_i) + 1e\text{-}8}$$

---

### 4.6 Bayesian Kernel Classification for Infeasible Regions

For experimental safety, conditions resulting in process failures or `NaN` outputs are encoded as $s_i = 1$ (infeasible) and valid runs as $s_i = 0$ (feasible).

Log-densities for feasible and infeasible spaces are estimated separately:
$$\log p(\mathbf{x}^* \mid \text{feasible}) = \log \left( \sum_{i: s_i=0} K(\mathbf{x}^*, \mathbf{x}_i) \right) - \log N_0$$
$$\log p(\mathbf{x}^* \mid \text{infeasible}) = \log \left( \sum_{i: s_i=1} K(\mathbf{x}^*, \mathbf{x}_i) \right) - \log N_1$$

Using Bayes' theorem, the posterior probability of feasibility $P(\text{feasible} \mid \mathbf{x}^*)$ is:
$$P(\text{feasible} \mid \mathbf{x}^*) = \frac{p(\mathbf{x}^* \mid \text{feasible}) \cdot \pi_0}{p(\mathbf{x}^* \mid \text{feasible}) \cdot \pi_0 + p(\mathbf{x}^* \mid \text{infeasible}) \cdot \pi_1}$$
where $\pi_0 = \frac{N_0}{N}$ and $\pi_1 = \frac{N_1}{N}$ are prior fractions.

---

### 4.7 Gryffin Acquisition Function & λ Parameter Dynamics

The acquisition function balances **exploitation** (minimizing predicted objective $\hat{y}$) and **exploration** (maximizing uncertainty in low-density regions):

$$\alpha(\mathbf{x}^*, \lambda) = \frac{\sum_{i \in \text{Feasible}} y_i \cdot K(\mathbf{x}^*, \mathbf{x}_i) + \lambda}{V^{-1} + \sum_{i \in \text{Feasible}} K(\mathbf{x}^*, \mathbf{x}_i)}$$

where $V = \prod_{k=1}^8 \Delta x_k$ is the search domain volume.

#### Sampling Strategies ($\lambda$)
- **$\lambda > 0$ (Exploitation Strategy)**: The numerator is dominated by $y_i \cdot K$. The acquisition function prefers regions with known low (optimal) objective values.
- **$\lambda < 0$ (Exploration Strategy)**: The numerator subtraction penalizes high density regions, driving candidates into unexplored domain spaces with low $\sum K$.

#### Feasibility Weighting (FWA)
When infeasible regions are present, the acquisition function is scaled by the probability of feasibility:
$$\alpha_{\text{FWA}}(\mathbf{x}^*) = -\left[ \left( 1 - \frac{\alpha(\mathbf{x}^*, \lambda) - \alpha_{\min}}{\alpha_{\max} - \alpha_{\min}} \right) \cdot P(\text{feasible} \mid \mathbf{x}^*) \right]$$

---

### 4.8 Finite-Difference Adam Optimizer with Constraints

Candidates drawn from uniform random sampling and incumbent perturbations are refined using Adam gradient descent on the acquisition function:

$$\nabla_{\mathbf{x}} \alpha(\mathbf{x}) \approx \frac{\alpha(\mathbf{x} + \delta \mathbf{e}_k) - \alpha(\mathbf{x} - \delta \mathbf{e}_k)}{2\delta}$$

#### Adam Update Rule
$$m^{(t)} = \beta_1 m^{(t-1)} + (1-\beta_1) \nabla_{\mathbf{x}} \alpha(\mathbf{x}^{(t)})$$
$$v^{(t)} = \beta_2 v^{(t-1)} + (1-\beta_2) \left(\nabla_{\mathbf{x}} \alpha(\mathbf{x}^{(t)})\right)^2$$
$$\mathbf{x}^{(t+1)} = \text{Proj}_{\Omega} \left[ \mathbf{x}^{(t)} - \eta \cdot \frac{m^{(t)}}{\sqrt{v^{(t)}} + \varepsilon} \right]$$

#### Constraint Projection Operator $\text{Proj}_{\Omega}$
If a proposed step $\mathbf{x}^{(t+1)}$ violates parameter bounds $[x_{\text{low}}, x_{\text{high}}]$ or user constraint function `known_constraints(x) == False`, the step is halted and projected back onto the nearest feasible boundary point.

---

### 4.9 Distance-Penalized Diversity Sample Selector

To generate batch recommendations of size $B$ without spatial clustering, candidate proposals are reweighted based on distance to existing observations and previously selected batch points:

$$\text{Reward}(\mathbf{x}^*) = \exp\left(-\alpha(\mathbf{x}^*)\right) \cdot \prod_{k=1}^8 \min\left(1.0, \exp\left(2 \cdot \frac{d_k - d_{\text{char}, k}}{\Delta x_k}\right)\right)$$

where:
- $d_k = \min_{j} |x_k^* - x_{j, k}|$ is the minimum distance to any observed or selected point along feature $k$.
- $d_{\text{char}, k} = \frac{\Delta x_k}{N^{0.5}}$ is the characteristic grid spacing.

If a candidate $\mathbf{x}^*$ is closer than $d_{\text{char}, k}$ to an existing point, its reward drops exponentially, guaranteeing diverse batch selection.

---

## 5. Step-by-Step Numerical Prediction & Optimization Walkthrough

Here is a step-by-step trace of how the algorithm processes data during one optimization loop:

```
[Input Data] 
N = 10 past experiments.
Features: 8 continuous variables.
Targets: 5 objective measurements.

   │
   ▼
[Step 1: Process Observations]
- Apply boundary reflection to points near domain edges.
- Scalarize 5 targets into single merit y_i ∈ [0, 1] using Chimera tolerances.
- Encode feasibility vector s_i ∈ {0, 1}.

   │
   ▼
[Step 2: Train BNN]
- Train PyTorch Variational Bayes MLP for 500 epochs.
- Sample S = 300 weight posterior draws.

   │
   ▼
[Step 3: Build Kernel Densities]
- Compute kernel centers μ and Gamma precisions τ.
- Evaluate Nadaraya-Watson kernel regression surrogate ŷ(x*).
- Evaluate Bayesian feasibility classifier P(feasible | x*).

   │
   ▼
[Step 4: Generate Candidates]
- Draw 200 uniform random samples & 200 incumbent perturbations.
- Select top candidates and run Adam gradient optimizer on α_FWA(x*, λ).
- Reject any candidates violating user constraint function g(x) <= 0.

   │
   ▼
[Step 5: Diversity Selection]
- Calculate exp(-α) acquisition rewards.
- Multiply rewards by spatial distance penalties.
- Return B optimal, diverse feature vectors for next experiment!
```

---

## 6. Code Component Walkthrough

| File | Primary Responsibility |
|---|---|
| `config.py` | Parses 8 continuous parameters, 5 objectives, tolerances, and BNN model details. |
| `observation_processor.py` | Implements boundary reflection and Chimera multi-objective scalarization. |
| `random_sampler.py` | Handles uniform sampling and incumbent perturbations with constraint rejection. |
| `bayesian_network/bnn_trainer.py` | PyTorch Variational Bayes MLP training and weight sampling. |
| `bayesian_network/numpy_graph.py` | Computes posterior draw forward graph (locations $\mu$, precisions $\tau$). |
| `bayesian_network/kernel_evaluator.py` | Vectorized NumPy Gaussian kernel evaluations and Nadaraya-Watson regression. |
| `bayesian_network/bayesian_network.py` | Coordinates kernel building, regression predictions, and feasibility checks. |
| `acquisition/adam_optimizer.py` | Finite-difference numerical gradient Adam optimizer with bound projections. |
| `acquisition/acquisition.py` | Calculates Gryffin acquisition function $\alpha(\mathbf{x}^*, \lambda)$ and generates candidates. |
| `sample_selector.py` | Distance-penalized reward selector for batch experiment recommendations. |
| `optimizer.py` | High-level user interface (`recommend()`, `get_regression_surrogate()`). |
| `run_example.py` | Fully functional 8-feature / 5-target runnable synthetic test script. |

---

## 7. Constraint Capability & Handling Infeasible Data

### Known Analytical Constraints
Users can pass custom domain constraints as Python functions:
```python
def my_physical_constraints(p):
    # p is a dictionary of feature values
    # Constraint 1: Max operating temperature x pressure
    if p["temperature"] * p["pressure"] > 1800.0:
        return False
    # Constraint 2: Minimum pH boundary
    if p["ph"] < 3.5:
        return False
    return True
```
This function is enforced during random sampling, candidate perturbation, and Adam optimization steps.

### Learned Feasibility Constraints
If an experiment failed in the lab (e.g., precipitation, decomposition, explosion risk), simply set target column values to `np.nan` or `None` in the observation dictionary:
```python
obs = {
    "temperature": 98.0, "ph": 11.5, "pressure": 25.0, ...
    "yield": np.nan, "purity": np.nan, "cost": np.nan, "byproduct": np.nan, "energy": np.nan
}
```
The optimizer automatically learns the infeasible boundary using Bayesian Kernel Classification and avoids proposing similar conditions.

---

## 8. Troubleshooting & Performance Tuning Guide

### Common Questions & Solutions

1. **How do I adjust exploration vs exploitation?**
   - Increase `sampling_strategies` in `config` (e.g., set to 3 or 4). This generates candidate batches using positive $\lambda$ (exploitation), zero $\lambda$, and negative $\lambda$ (exploration).

2. **What if optimization is slow?**
   - For quick exploratory runs, reduce `num_epochs` to `200` and `num_draws` to `100` in model config.
   - For final production optimization, use `num_epochs = 1000` and `num_draws = 500`.

3. **How do I change target priorities?**
   - In the `objectives` list, place the most critical target first (e.g. `yield`), followed by secondary targets. Adjust `tolerance` values (e.g. `0.05` means 5% tolerance before lower-priority objectives take over).

4. **Multi-CPU Acceleration**:
   - Set `"num_cpus": "all"` in the general config to parallelize random sampling and Adam acquisition optimization across all available CPU cores.
