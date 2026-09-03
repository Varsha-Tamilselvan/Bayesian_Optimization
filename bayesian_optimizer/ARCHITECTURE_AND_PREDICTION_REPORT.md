# Architectural & Mathematical Specification Report
## Continuous Multi-Objective Bayesian Optimization Model (Gryffin Architecture)

---

## Executive Summary

This report provides a comprehensive mathematical and architectural documentation of the customized **Gryffin-inspired Bayesian Optimization Model**. Designed specifically for continuous experimental design domains, this framework optimizes **8 continuous process features** (e.g., temperature, pH, pressure, concentrations, flow rate, time, agitation speed) to maximize/minimize **5 target objectives** while strictly enforcing **physical domain constraints** and accounting for **infeasible experimental observations**.

Unlike traditional Gaussian Process (GP) Bayesian optimization, which scales cubically $\mathcal{O}(N^3)$ with dataset size $N$, this model leverages **Linear-Scaling Kernel Regression via Bayesian Neural Networks (BNNs)**, scaling linearly $\mathcal{O}(N)$.

---

## 1. System Architecture Overview

```
                      +-----------------------------------+
                      |   Historical Experimental Data    |
                      |  (8 Features -> 5 Target Columns) |
                      +-----------------+-----------------+
                                        |
                                        v
                      +-----------------------------------+
                      |   1. Observation Processing       |
                      |  - Parameter Boundary Mirroring   |
                      |  - Chimera Multi-Obj Scalarizer   |
                      |  - Feasibility Vector Assignment  |
                      +-----------------+-----------------+
                                        |
                                        v
                      +-----------------------------------+
                      |   2. Bayesian Neural Network      |
                      |  - Variational Bayes MLP Training |
                      |  - ELBO = NLL + KL Loss           |
                      |  - Posterior Draw Extraction      |
                      +-----------------+-----------------+
                                        |
                                        v
                      +-----------------------------------+
                      |   3. Kernel Density Estimation    |
                      |  - Gaussian Location Estimates    |
                      |  - Gamma Precision Estimates (τ)  |
                      |  - Nadaraya-Watson Regression     |
                      |  - Bayesian Feasibility Classifier|
                      +-----------------+-----------------+
                                        |
                                        v
                      +-----------------------------------+
                      |   4. Acquisition Optimization     |
                      |  - λ-Strategies (Exploit/Explore) |
                      |  - Rejection Sampling + Perturb   |
                      |  - Finite-Diff Adam Optimization  |
                      |  - Constraint Verification        |
                      +-----------------+-----------------+
                                        |
                                        v
                      +-----------------------------------+
                      |   5. Diversity Sample Selector    |
                      |  - Exp(-Acq) Reward Calculation   |
                      |  - Distance Penalty Reweighting   |
                      |  - Final Candidate Selection      |
                      +-----------------+-----------------+
                                        |
                                        v
                      +-----------------------------------+
                      | Recommended Experimental Features |
                      +-----------------------------------+
```

---

## 2. Detailed Component Breakdown & Prediction Workflow

### 2.1 Observation Processor & Multi-Objective Scalarization

#### A. Parameter Boundary Mirroring
To eliminate kernel edge-effects near parameter boundaries $[x_{\text{low}}, x_{\text{high}}]$, observations located within the outer 10% region of the bounds are reflected across domain limits.
$$\tilde{x}_{\text{lower}} = x_{\text{low}} - (x - x_{\text{low}}), \quad \tilde{x}_{\text{upper}} = x_{\text{high}} + (x_{\text{high}} - x)$$

#### B. Hierarchical Multi-Objective Chimera Scalarization
The 5 raw target objectives $\mathbf{y} = [y_1, y_2, y_3, y_4, y_5]$ are scalarized into a unified merit value $f(x) \in [0, 1]$, where $0$ represents the optimal performance and $1$ represents the worst.

1. **Normalization**: Each objective $y_i$ is mapped to $z_i \in [0, 1]$ based on whether it is to be maximized or minimized:
   $$z_i = \begin{cases} \frac{y_i - \min(y_i)}{\max(y_i) - \min(y_i)} & \text{if minimizing} \\[6pt] 1 - \frac{y_i - \min(y_i)}{\max(y_i) - \min(y_i)} & \text{if maximizing} \end{cases}$$

2. **Smooth Soft Thresholding**: Objectives are folded hierarchically according to user tolerances $\tau_i$ using a smooth activation:
   $$w_i = \frac{1}{2} \left[ 1 + \tanh\left(\frac{z_i - \tau_i}{\varepsilon}\right) \right]$$
   $$m_i = w_i \cdot z_i + (1 - w_i) \cdot m_{i+1}$$

#### C. Infeasibility Encoding
Observations with missing values (`NaN`), process failures, or constraint violations are marked with feasibility flag $s_i = 1.0$ (infeasible) vs $s_i = 0.0$ (feasible).

---

### 2.2 Bayesian Neural Network (BNN) Surrogate

#### A. Variational Inference Architecture
The BNN models the distribution over model parameters $w \sim q_\theta(w) = \mathcal{N}(\mu, \sigma^2)$ rather than deterministic point estimates.
- **Layers**: 3-Layer MLP with ReLU hidden activations.
- **Inputs**: Rescaled feature vector $\mathbf{x}_{\text{norm}} = \frac{\mathbf{x} - \mathbf{x}_{\text{low}}}{\mathbf{x}_{\text{high}} - \mathbf{x}_{\text{low}}} \in [0, 1]^8$.
- **Outputs**: Latent kernel parameter distributions.

#### B. Loss Function Optimization (ELBO)
The BNN is trained by minimizing the negative Evidence Lower Bound (ELBO):
$$\mathcal{L}(\theta) = -\sum_{i=1}^N \log p\left(y_i \mid \mathbf{x}_i, w\right) + \frac{1}{N} \text{KL}\left( q_\theta(w) \parallel p(w) \right)$$
where $p(w) = \mathcal{N}(0, I)$ is the Gaussian prior.

#### C. Posterior Sampling
After training, $S = 300\text{--}1000$ weight sets are sampled from $q_\theta(w)$ and passed through the forward graph to generate joint kernel locations $\mu_{s, i, k}$ and precision parameters $\tau_{s, i, k} \sim \text{Gamma}(\alpha, \beta)$.

---

### 2.3 Nadaraya-Watson Gaussian Kernel Density Evaluator

#### A. Gaussian Kernel Density Computation
For any query feature point $\mathbf{x}^* \in \mathbb{R}^8$, the kernel probability $K(\mathbf{x}^*, \mathbf{x}_i)$ relative to observation $i$ is calculated by averaging over BNN posterior samples:
$$K(\mathbf{x}^*, \mathbf{x}_i) = \frac{1}{S} \sum_{s=1}^S \prod_{k=1}^D \frac{\sqrt{\tau_{s, i, k}}}{\sqrt{2\pi}} \exp\left( -\frac{1}{2} \tau_{s, i, k} \left( x_k^* - \mu_{s, i, k} \right)^2 \right)$$

#### B. Nadaraya-Watson Regression Prediction
The predicted scalarized objective $\hat{y}(\mathbf{x}^*)$ at unmeasured location $\mathbf{x}^*$ is given by:
$$\hat{y}(\mathbf{x}^*) = \frac{\sum_{i=1}^{N_{\text{feas}}} y_i \cdot K(\mathbf{x}^*, \mathbf{x}_i)}{\sum_{i=1}^{N_{\text{feas}}} K(\mathbf{x}^*, \mathbf{x}_i) + 1e\text{-}8}$$

#### C. Bayesian Feasibility Posterior Prediction
The probability that a condition $\mathbf{x}^*$ is physically feasible is computed via Bayes' theorem:
$$P(\text{feasible} \mid \mathbf{x}^*) = \frac{p(\mathbf{x}^* \mid \text{feasible}) \cdot \pi_0}{p(\mathbf{x}^* \mid \text{feasible}) \cdot \pi_0 + p(\mathbf{x}^* \mid \text{infeasible}) \cdot \pi_1}$$
where $\pi_0, \pi_1$ are the prior fractions of feasible and infeasible historical observations.

---

### 2.4 Acquisition Function & Multi-Strategy Optimization

#### A. Exploitation vs. Exploration Balance ($\lambda$)
Gryffin uses an acquisition parameter $\lambda$ to balance exploitation (sampling near known optima) vs exploration (sampling uncertain regions):
$$\alpha(\mathbf{x}^*, \lambda) = \frac{\sum_{i=1}^{N_{\text{feas}}} y_i \cdot K(\mathbf{x}^*, \mathbf{x}_i) + \lambda}{V^{-1} + \sum_{i=1}^{N_{\text{feas}}} K(\mathbf{x}^*, \mathbf{x}_i)}$$
where $V = \prod_{k=1}^8 (x_{\text{high}, k} - x_{\text{low}, k})$ is the search domain volume.
- $\lambda > 0$: **Exploitation** (bias towards low $y_i$).
- $\lambda < 0$: **Exploration** (bias towards low kernel density $\sum K$).

#### B. Feasibility-Weighted Acquisition (FWA)
When infeasible regions exist, the acquisition function is scaled by feasibility probability:
$$\alpha_{\text{FWA}}(\mathbf{x}^*) = -\left[ (1 - \tilde{\alpha}(\mathbf{x}^*, \lambda)) \cdot P(\text{feasible} \mid \mathbf{x}^*) \right]$$

#### C. Finite-Difference Adam Optimizer
Candidates are drawn via uniform random sampling and perturbation around the current incumbent, then refined using Adam gradient descent:
$$\mathbf{x}^{(t+1)} = \text{Proj}_{\Omega} \left[ \mathbf{x}^{(t)} - \eta \cdot \frac{m^{(t)}}{\sqrt{v^{(t)}} + \varepsilon} \right]$$
where $\text{Proj}_{\Omega}$ projects points back into bounds and enforces user constraints `known_constraints(x) == True`.

---

### 2.5 Diversity-Aware Sample Selector

To avoid recommending batch experiments that are clustered together, proposals are penalised based on Euclidean distance to existing measurements and previously selected batch points:

$$\text{Reward}(\mathbf{x}^*) = \exp\left(-\alpha(\mathbf{x}^*)\right) \cdot \prod_{k=1}^8 \min\left(1.0, \exp\left(2 \cdot \frac{d_k - d_{\text{char}, k}}{\Delta x_k}\right)\right)$$

where:
- $d_k = \min_{j} |x_k^* - x_{j, k}|$ is the minimum distance along dimension $k$.
- $d_{\text{char}, k} = \frac{\Delta x_k}{N^{0.5}}$ is the characteristic point spacing.

---

## 3. How Prediction Works Step-by-Step

| Step | Operation | Output |
|---|---|---|
| **1. Collect Input Data** | Load observed experimental features & target columns | Feature matrix $\mathbf{X} \in \mathbb{R}^{N \times 8}$, Target matrix $\mathbf{Y} \in \mathbb{R}^{N \times 5}$ |
| **2. Scalarize Targets** | Chimera hierarchical folding + boundary mirroring | Scalarized merit $y \in [0, 1]^N$, Feasibility flags $s \in \{0, 1\}^N$ |
| **3. Train BNN** | Optimize ELBO loss via PyTorch Adam | Posterior weight distributions $q_\theta(w)$ |
| **4. Posterior Sampling** | Draw $S$ weight sets & evaluate NumPy Graph | Kernel centers $\mathbf{\mu} \in \mathbb{R}^{S \times N \times 8}$, Precisions $\mathbf{\tau} \in \mathbb{R}^{S \times N \times 8}$ |
| **5. Query Predictions** | Evaluate Nadaraya-Watson regression & feasibility | Target estimation $\hat{y}(\mathbf{x}^*)$, Feasibility probability $P(\text{feas} \mid \mathbf{x}^*)$ |
| **6. Optimize Acquisition** | Run Adam on $\alpha(\mathbf{x}^*, \lambda)$ with constraint checks | Candidate pool of optimized proposals |
| **7. Select Batch** | Apply distance penalty reweighting | Recommended experimental feature vectors |

---

## 4. Verification and Practical Usage

The complete architecture has been implemented in standard Python/PyTorch/NumPy and verified:
- **No C-extension dependencies**: Completely portable across Windows, Linux, and macOS.
- **Full Constraint Support**: Supports arbitrary non-linear user functions (e.g. `temp * pressure <= 1800`).
- **Runnable Demo**: Execute `python -m bayesian_optimizer.run_example` to verify model performance on synthetic experimental surfaces.
