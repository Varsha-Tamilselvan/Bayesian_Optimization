import numpy as np
from ..observation_processor import param_vector_to_dict

class AdamOptimizer:
    """
    Adam Optimizer with finite-difference gradient estimation and bound projections.
    """
    def __init__(self, eta=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.eta = eta
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.dx = 1e-5
        self.reset()

    def reset(self):
        self.iterations = 0
        self.ms = None
        self.vs = None

    def grad(self, func, sample):
        gradients = np.zeros_like(sample, dtype=np.float64)
        perturb = np.zeros_like(sample, dtype=np.float64)

        for i in range(len(sample)):
            perturb[i] += self.dx
            val_plus = func(sample + perturb)
            val_minus = func(sample - perturb)
            gradients[i] = (val_plus - val_minus) / (2.0 * self.dx)
            perturb[i] -= self.dx

        return gradients

    def get_update(self, func, sample):
        if self.ms is None:
            self.ms = np.zeros_like(sample, dtype=np.float64)
            self.vs = np.zeros_like(sample, dtype=np.float64)

        grads = self.grad(func, sample)
        self.iterations += 1

        eta_next = self.eta * (np.sqrt(1.0 - np.power(self.beta_2, self.iterations)) /
                              (1.0 - np.power(self.beta_1, self.iterations) + 1e-10))

        self.ms = (self.beta_1 * self.ms) + (1.0 - self.beta_1) * grads
        self.vs = (self.beta_2 * self.vs) + (1.0 - self.beta_2) * np.square(grads)

        sample_next = sample - eta_next * self.ms / (np.sqrt(self.vs) + self.epsilon)
        return sample_next


class GradientOptimizer:
    """
    Gradient-based optimizer for acquisition function optimization with constraint checking.
    """
    def __init__(self, config, constraints=None):
        self.config = config
        if constraints is not None and not isinstance(constraints, list):
            self.constraints = [constraints]
        else:
            self.constraints = constraints or []

        self.adam = AdamOptimizer()

    def _within_bounds(self, sample):
        return not (np.any(sample < self.config.param_lowers) or np.any(sample > self.config.param_uppers))

    def _project_sample(self, sample):
        return np.clip(sample, self.config.param_lowers, self.config.param_uppers)

    def _is_feasible(self, sample):
        if not self._within_bounds(sample):
            return False
        if not self.constraints:
            return True

        param_dict = param_vector_to_dict(sample, self.config.param_names)
        for constr in self.constraints:
            try:
                res = constr(param_dict)
            except Exception:
                res = constr(sample)
            if not res:
                return False
        return True

    def optimize(self, func, samples, max_iter=10):
        optimized_samples = []

        for sample in samples:
            self.adam.reset()
            current = sample.copy()
            best_val = func(current)
            best_sample = current.copy()

            for _ in range(max_iter):
                proposal = self.adam.get_update(func, current)
                proposal = self._project_sample(proposal)

                if self._is_feasible(proposal):
                    val = func(proposal)
                    if val < best_val:  # Minimizing acquisition function
                        best_val = val
                        best_sample = proposal.copy()
                    current = proposal
                else:
                    break

            optimized_samples.append(best_sample)

        return np.array(optimized_samples, dtype=np.float64)
