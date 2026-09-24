"""
Bioprocess Multi-Objective Bayesian Optimization Framework from Scratch
-----------------------------------------------------------------------
Featuring:
1. Continuous Data Scaling (7 Input Variables to [0, 1]^7)
2. Chimera Achievement Scalarizing Function (ASF) (5 Target Variables -> Single Scalar)
3. PHOENICS/GRYFFIN Kernel Center & Precision-Based Objective Model (tau_n = 12 * n^2)
4. Gaussian Process (GP) Surrogate Model
5. Acquisition Sampler with Exploration/Exploitation Parameter lambda in [-1, 1]
6. Closed-Loop Optimization Pipeline & Model Update Code

Inputs (7 Continuous Variables):
  - initial_vcc, initial_temp, temp_shift, do, cb7a, cb7b, production_media

Targets (5 Continuous Variables):
  - max_vcc, harvest_vcc, harvest_viability, enzyme_activity, ivcc
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from typing import Dict, List, Tuple, Union, Optional


# ==============================================================================
# 1. CONTINUOUS DATA SCALER
# ==============================================================================

class ContinuousDataScaler:
    """Scales continuous input features to [0, 1]^d hypercube and inverse-scales back."""
    def __init__(self, feature_bounds: Dict[str, Tuple[float, float]]):
        self.feature_names = list(feature_bounds.keys())
        self.feature_bounds = feature_bounds
        self.lowers = np.array([feature_bounds[f][0] for f in self.feature_names], dtype=np.float64)
        self.uppers = np.array([feature_bounds[f][1] for f in self.feature_names], dtype=np.float64)
        self.ranges = self.uppers - self.lowers
        self.ranges[self.ranges == 0.0] = 1.0

    def transform(self, df_or_arr: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(df_or_arr, pd.DataFrame):
            arr = df_or_arr[self.feature_names].values.astype(np.float64)
        else:
            arr = np.asarray(df_or_arr, dtype=np.float64)
        scaled = (arr - self.lowers) / self.ranges
        return np.clip(scaled, 0.0, 1.0)

    def inverse_transform(self, scaled_arr: np.ndarray) -> pd.DataFrame:
        scaled_arr = np.asarray(scaled_arr, dtype=np.float64)
        if scaled_arr.ndim == 1:
            scaled_arr = scaled_arr.reshape(1, -1)
        raw = scaled_arr * self.ranges + self.lowers
        return pd.DataFrame(raw, columns=self.feature_names)


# ==============================================================================
# 2. CHIMERA MULTI-OBJECTIVE TARGET SCALARIZER (Hase et al., 2018)
# ==============================================================================

class ChimeraScalarizer:
    """
    Hierarchical Achievement Scalarizing Function (ASF) implementing Chimera.
    Converts 5 multi-objective targets into 1 single scalar value chi(x) in [0, 1].
    """
    def __init__(
        self,
        hierarchy: List[str],
        goals: Dict[str, str],
        tolerances: Dict[str, float],
        softness: float = 0.001,
        absolute_tolerances: Optional[Dict[str, float]] = None
    ):
        self.hierarchy = hierarchy
        self.goals = goals
        self.tolerances = tolerances
        self.softness = softness
        self.absolute_tolerances = absolute_tolerances or {}

    def _logistic_step(self, val: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        diff = (val - threshold) / self.softness
        diff_clipped = np.clip(diff, -50.0, 50.0)
        theta_plus = 1.0 / (1.0 + np.exp(diff_clipped))
        theta_minus = 1.0 - theta_plus
        return theta_plus, theta_minus

    def scalarize(self, df_targets: pd.DataFrame) -> np.ndarray:
        num_obs = len(df_targets)
        num_objs = len(self.hierarchy)
        if num_obs == 0:
            return np.array([])

        norm_objs = np.zeros((num_obs, num_objs), dtype=np.float64)
        abs_tols = np.zeros(num_objs, dtype=np.float64)

        for i, obj_name in enumerate(self.hierarchy):
            raw_vals = df_targets[obj_name].values.astype(np.float64)
            goal = self.goals.get(obj_name, 'max').lower()
            
            g_vals = -raw_vals if goal == 'max' else raw_vals
            g_min, g_max = np.min(g_vals), np.max(g_vals)
            g_range = g_max - g_min if (g_max - g_min) != 0.0 else 1.0

            norm_objs[:, i] = (g_vals - g_min) / g_range
            abs_tols[i] = self.tolerances.get(obj_name, 0.1)

        theta_plus = np.zeros((num_obs, num_objs), dtype=np.float64)
        theta_minus = np.zeros((num_obs, num_objs), dtype=np.float64)

        for i in range(num_objs):
            t_plus, t_minus = self._logistic_step(norm_objs[:, i], abs_tols[i])
            theta_plus[:, i] = t_plus
            theta_minus[:, i] = t_minus

        shifts = np.zeros(num_objs, dtype=np.float64)
        for i in range(num_objs):
            unsatisfied_mask = theta_minus[:, i] > 0.5
            shifts[i] = np.min(norm_objs[unsatisfied_mask, i]) if np.any(unsatisfied_mask) else np.min(norm_objs[:, i])

        chi = norm_objs[:, 0] * theta_plus[:, 0]
        chi += (norm_objs[:, 0] - shifts[-1]) * np.prod(theta_minus, axis=1)

        for k in range(1, num_objs):
            prod_minus_prev = np.prod(theta_minus[:, :k], axis=1)
            chi += (norm_objs[:, k] - shifts[k - 1]) * theta_plus[:, k] * prod_minus_prev

        chi_min, chi_max = np.min(chi), np.max(chi)
        chi_range = chi_max - chi_min
        return (chi - chi_min) / chi_range if chi_range > 1e-12 else np.zeros_like(chi)


# ==============================================================================
# 3. KERNEL CENTER & PRECISION-BASED BNN SURROGATE MODEL (PHOENICS & GRYFFIN)
# ==============================================================================

class BNNKernelCenterNet(nn.Module):
    """Deep Neural Network mapping observation features x_k to Kernel Centers mu_k."""
    def __init__(self, dim: int = 7, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid()  # Keep kernel centers bounded in [0, 1]^d
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KernelPrecisionSurrogateModel:
    """
    Exact PHOENICS / GRYFFIN Kernel Center & Precision-Based Objective Model (Hase et al., 2018).
    
    1. Kernel Centers mu_k:
       Mapped from observations x_k using Neural Net phi(x_k; theta).
    2. Precision Schedule tau_n:
       tau_n = 12 * n^2 (shrinks Gaussian variance as observation count n grows).
    3. Kernel Density p_k(x):
       p_k(x) = (tau_n / (2*pi))^(d/2) * exp( - (tau_n / 2) * ||x - mu_k||^2 )
    4. Precision-based Objective Approximation alpha(x):
       alpha(x) = sum_k( f_k * p_k(x) ) / sum_k( p_k(x) )
    5. Acquisition Function alpha_lambda(x):
       alpha_lambda(x) = ( sum_k( f_k * p_k(x) ) + lambda * p_uniform(x) ) / ( sum_k( p_k(x) ) + p_uniform(x) )
    """
    def __init__(self, dim: int = 7, lr: float = 0.01, epochs: int = 150):
        self.dim = dim
        self.lr = lr
        self.epochs = epochs
        self.model = BNNKernelCenterNet(dim=dim)
        self.kernel_centers = None
        self.y_obs = None
        self.n_obs = 0
        self.tau_n = 12.0

    def fit(self, X_scaled: np.ndarray, y_scalarized: np.ndarray):
        """Fits kernel centers and updates precision schedule tau_n = 12 * n^2."""
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scalarized, dtype=torch.float32).view(-1, 1)

        self.n_obs = len(X_scaled)
        # Precision schedule scaling quadratically with observations: tau_n = 12 * n^2
        self.tau_n = 12.0 * (self.n_obs ** 2)

        # Train BNN/Neural Net to map observation locations to optimal kernel centers
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            predicted_centers = self.model(X_tensor)
            loss = criterion(predicted_centers, X_tensor)
            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            self.kernel_centers = self.model(X_tensor).numpy()
        self.y_obs = y_scalarized.ravel()

    def evaluate_kernel_densities(self, X_query: np.ndarray) -> np.ndarray:
        """
        Computes Gaussian kernel densities p_k(x) for all query points x against observation kernel centers mu_k.
        p_k(x) = (tau_n / (2*pi))^(d/2) * exp( - (tau_n / 2) * ||x - mu_k||^2 )
        """
        X_query = np.asarray(X_query, dtype=np.float64)
        if X_query.ndim == 1:
            X_query = X_query.reshape(1, -1)

        n_query = len(X_query)
        p_k = np.zeros((n_query, self.n_obs), dtype=np.float64)

        # Prefactor for d-dimensional Gaussian precision
        prefactor = (self.tau_n / (2.0 * np.pi)) ** (self.dim / 2.0)

        for k in range(self.n_obs):
            diff = X_query - self.kernel_centers[k]  # (n_query, dim)
            sq_dist = np.sum(diff ** 2, axis=1)      # (n_query,)
            exponent = -0.5 * self.tau_n * sq_dist
            exponent_clipped = np.clip(exponent, -50.0, 50.0)
            p_k[:, k] = prefactor * np.exp(exponent_clipped)

        return p_k

    def predict_objective_landscape(self, X_query: np.ndarray) -> np.ndarray:
        """
        Computes precision-based objective function approximation alpha(x):
        alpha(x) = sum_k( f_k * p_k(x) ) / sum_k( p_k(x) )
        """
        p_k = self.evaluate_kernel_densities(X_query)
        numerator = np.sum(self.y_obs * p_k, axis=1)
        denominator = np.sum(p_k, axis=1)
        
        # Avoid division by zero in unobserved regions
        mask = denominator < 1e-12
        denominator[mask] = 1.0
        numerator[mask] = np.mean(self.y_obs)

        return numerator / denominator

    def acquisition_function(self, X_query: np.ndarray, sampling_param_lambda: float = 0.0) -> np.ndarray:
        """
        Computes PHOENICS acquisition function alpha_lambda(x):
        alpha_lambda(x) = ( sum_k( f_k * p_k(x) ) + lambda * p_uniform ) / ( sum_k( p_k(x) ) + p_uniform )
        p_uniform = 1.0 over unit hypercube [0, 1]^d.
        """
        p_k = self.evaluate_kernel_densities(X_query)
        numerator = np.sum(self.y_obs * p_k, axis=1) + sampling_param_lambda * 1.0
        denominator = np.sum(p_k, axis=1) + 1.0
        return numerator / denominator


# ==============================================================================
# 4. GAUSSIAN PROCESS SURROGATE MODEL (ALTERNATIVE REGRESSOR)
# ==============================================================================

class GPSurrogateModel:
    """Standard Gaussian Process Surrogate Model mapping scaled inputs X to scalarized target y."""
    def __init__(self, random_state: int = 42):
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(7), length_scale_bounds=(1e-2, 1e2), nu=2.5) \
                 + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, random_state=random_state, alpha=1e-6)

    def fit(self, X_scaled: np.ndarray, y_scalarized: np.ndarray):
        self.gp.fit(np.asarray(X_scaled, dtype=np.float64), np.asarray(y_scalarized, dtype=np.float64).ravel())

    def predict(self, X_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = np.asarray(X_scaled, dtype=np.float64)
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(1, -1)
        mu, std = self.gp.predict(X_scaled, return_std=True)
        return mu, np.maximum(std, 1e-8)


# ==============================================================================
# 5. KERNEL DENSITY & ACQUISITION SAMPLER
# ==============================================================================

class AcquisitionOptimizer:
    """Optimizes Acquisition Functions (Kernel Precision alpha_lambda or GP LCB/EI) to suggest candidate experiments."""
    def __init__(self, dim: int = 7, random_state: int = 42):
        self.dim = dim
        self.rng = np.random.RandomState(random_state)

    def suggest_next_points_kernel_precision(
        self,
        kernel_model: KernelPrecisionSurrogateModel,
        batch_size: int = 1,
        n_restarts: int = 15
    ) -> np.ndarray:
        """
        Proposes candidate points by minimizing PHOENICS acquisition alpha_lambda(x)
        across a batch of sampling parameters lambda in [-1, 1].
        lambda > 0: Exploitation bias
        lambda < 0: Exploration bias
        """
        bounds = [(0.0, 1.0)] * self.dim
        suggested_points = []
        lambdas = np.linspace(-0.8, 0.8, batch_size) if batch_size > 1 else [0.0]

        for b in range(batch_size):
            lambda_b = lambdas[b]
            def obj_func(x):
                return kernel_model.acquisition_function(x.reshape(1, -1), sampling_param_lambda=lambda_b)[0]

            start_points = self.rng.uniform(0.0, 1.0, size=(n_restarts, self.dim))
            best_x, best_val = None, np.inf

            for start in start_points:
                res = minimize(obj_func, x0=start, bounds=bounds, method='L-BFGS-B')
                if res.fun < best_val:
                    best_val, best_x = res.fun, res.x

            suggested_points.append(best_x)

        return np.array(suggested_points)


# ==============================================================================
# 6. CLOSED-LOOP BAYESIAN OPTIMIZER (WITH KERNEL PRECISION & CHIMERA)
# ==============================================================================

class BioprocessBayesianOptimizer:
    """Full End-to-End Multi-Objective BO Pipeline using Kernel Precision & Chimera."""
    def __init__(
        self,
        feature_bounds: Dict[str, Tuple[float, float]],
        hierarchy: List[str],
        goals: Dict[str, str],
        tolerances: Dict[str, float],
        softness: float = 0.001,
        surrogate_type: str = 'kernel_precision',  # 'kernel_precision' or 'gp'
        random_state: int = 42
    ):
        self.scaler = ContinuousDataScaler(feature_bounds)
        self.scalarizer = ChimeraScalarizer(hierarchy, goals, tolerances, softness)
        self.surrogate_type = surrogate_type
        
        if surrogate_type == 'kernel_precision':
            self.surrogate = KernelPrecisionSurrogateModel(dim=len(feature_bounds))
        else:
            self.surrogate = GPSurrogateModel(random_state)

        self.acq_optimizer = AcquisitionOptimizer(dim=len(feature_bounds), random_state=random_state)
        
        self.X_history_raw = pd.DataFrame(columns=self.scaler.feature_names)
        self.Y_history_raw = pd.DataFrame(columns=hierarchy)
        self.X_history_scaled = np.empty((0, len(feature_bounds)))
        self.y_chimera_history = np.array([])

    def add_observations(self, X_df: pd.DataFrame, Y_df: pd.DataFrame):
        if self.X_history_raw.empty:
            self.X_history_raw = X_df[self.scaler.feature_names].copy().reset_index(drop=True)
            self.Y_history_raw = Y_df[self.scalarizer.hierarchy].copy().reset_index(drop=True)
        else:
            self.X_history_raw = pd.concat([self.X_history_raw, X_df[self.scaler.feature_names]], ignore_index=True)
            self.Y_history_raw = pd.concat([self.Y_history_raw, Y_df[self.scalarizer.hierarchy]], ignore_index=True)

        self.X_history_scaled = self.scaler.transform(self.X_history_raw)
        self.y_chimera_history = self.scalarizer.scalarize(self.Y_history_raw)

    def suggest_next_experiments(self, batch_size: int = 1) -> pd.DataFrame:
        self.surrogate.fit(self.X_history_scaled, self.y_chimera_history)

        if self.surrogate_type == 'kernel_precision':
            next_scaled = self.acq_optimizer.suggest_next_points_kernel_precision(self.surrogate, batch_size=batch_size)
        else:
            # GP fallback
            bounds = [(0.0, 1.0)] * self.scaler.lowers.shape[0]
            start_points = np.random.uniform(0.0, 1.0, size=(15, len(bounds)))
            best_x, best_val = None, np.inf
            for start in start_points:
                res = minimize(lambda x: self.surrogate.predict(x.reshape(1, -1))[0][0], x0=start, bounds=bounds, method='L-BFGS-B')
                if res.fun < best_val:
                    best_val, best_x = res.fun, res.x
            next_scaled = np.array([best_x])

        return self.scaler.inverse_transform(next_scaled)

    def step(self, evaluator_fn, batch_size: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_next_df = self.suggest_next_experiments(batch_size=batch_size)
        Y_next_df = evaluator_fn(X_next_df)
        self.add_observations(X_next_df, Y_next_df)
        return X_next_df, Y_next_df


# ==============================================================================
# 7. SIMULATED BIOPROCESS SYSTEM FOR DEMONSTRATION & VERIFICATION
# ==============================================================================

def simulated_bioprocess_evaluator(X_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for _, row in X_df.iterrows():
        vcc0, temp0, t_shift, do, cb7a, cb7b, media = (
            row['initial_vcc'], row['initial_temp'], row['temp_shift'],
            row['do'], row['cb7a'], row['cb7b'], row['production_media']
        )
        temp_opt, do_opt, media_opt = 35.5 - 0.2 * t_shift, 45.0, 85.0
        growth_rate = max(0.03 * media + 0.15 * cb7a - 0.08 * (temp0 - temp_opt)**2 - 0.005 * (do - do_opt)**2, 0.1)

        max_vcc = 5.0 + growth_rate * 2.5 + 1.2 * vcc0 + np.random.normal(0, 0.2)
        harvest_vcc = max_vcc * (0.75 + 0.03 * cb7b - 0.02 * t_shift) + np.random.normal(0, 0.15)
        harvest_viability = np.clip(95.0 - 1.8 * (temp0 - 34.0) - 2.0 * t_shift + 0.5 * cb7b + np.random.normal(0, 0.5), 50.0, 99.0)
        enzyme_activity = (100.0 + 15.0 * cb7a + 12.0 * cb7b - 0.5 * (media - media_opt)**2) * (harvest_viability / 100.0) + np.random.normal(0, 2.0)
        ivcc = (max_vcc + harvest_vcc) * 7.0 + np.random.normal(0, 1.0)

        results.append({
            'max_vcc': float(max_vcc), 'harvest_vcc': float(harvest_vcc),
            'harvest_viability': float(harvest_viability),
            'enzyme_activity': float(enzyme_activity), 'ivcc': float(ivcc)
        })
    return pd.DataFrame(results)


# ==============================================================================
# MAIN EXECUTION TEST
# ==============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("BIOPROCESS BAYESIAN OPTIMIZATION WITH KERNEL PRECISION & CHIMERA ASF")
    print("=" * 80)

    feature_bounds = {
        'initial_vcc': (0.2, 2.0), 'initial_temp': (33.0, 37.0), 'temp_shift': (0.0, 5.0),
        'do': (20.0, 60.0), 'cb7a': (1.0, 10.0), 'cb7b': (0.5, 5.0), 'production_media': (50.0, 100.0)
    }
    hierarchy = ['max_vcc', 'harvest_viability', 'enzyme_activity', 'harvest_vcc', 'ivcc']
    goals = {'max_vcc': 'max', 'harvest_viability': 'max', 'enzyme_activity': 'max', 'harvest_vcc': 'max', 'ivcc': 'max'}
    tolerances = {'max_vcc': 0.10, 'harvest_viability': 0.05, 'enzyme_activity': 0.10, 'harvest_vcc': 0.15, 'ivcc': 0.20}

    optimizer = BioprocessBayesianOptimizer(
        feature_bounds=feature_bounds, hierarchy=hierarchy, goals=goals,
        tolerances=tolerances, softness=0.001, surrogate_type='kernel_precision'
    )

    np.random.seed(42)
    initial_X = {feat: np.random.uniform(low, high, size=10) for feat, (low, high) in feature_bounds.items()}
    X_init_df = pd.DataFrame(initial_X)
    Y_init_df = simulated_bioprocess_evaluator(X_init_df)

    optimizer.add_observations(X_init_df, Y_init_df)
    print(f"\n[Step 1] Initialized {len(X_init_df)} initial observations.")

    print("\n[Step 2] Running 3 Closed-Loop Bayesian Optimization Iterations...")
    for iter_idx in range(3):
        print(f"\n--- Iteration {iter_idx + 1}/3 ---")
        X_next, Y_next = optimizer.step(simulated_bioprocess_evaluator, batch_size=2)
        print("Suggested Experiments (Raw Physical Units):")
        print(X_next.round(3))
        print("Evaluated Multi-Objective Target Results:")
        print(Y_next.round(3))

    best_idx = np.argmin(optimizer.y_chimera_history)
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE - BEST EXPERIMENTAL CONDITION FOUND")
    print("=" * 80)
    print("Best Input Parameters:", optimizer.X_history_raw.iloc[best_idx].to_dict())
    print("Best Target Values:", optimizer.Y_history_raw.iloc[best_idx].to_dict())
    print(f"Best Chimera Scalar Value: {optimizer.y_chimera_history[best_idx]:.4f}")
