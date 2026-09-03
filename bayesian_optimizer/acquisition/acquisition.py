import numpy as np
from ..random_sampler import RandomSampler
from .adam_optimizer import GradientOptimizer
from ..observation_processor import param_vector_to_dict

class AcquisitionFunction:
    """
    Evaluates the Gryffin acquisition function given a sampling parameter λ.
    Supports 3 feasibility handling modes:
      - 'fwa': Feasibility-Weighted Acquisition = Acq(x) * P(feasible | x)
      - 'fia': Feasibility-Interpolated Acquisition = k * P(infeasible | x) + (1-k) * Acq(x)
      - 'fca': Feasibility-Constrained Acquisition = Acq(x) subject to P(feasible | x) >= threshold
    """
    def __init__(self, bayesian_network, sampling_param, acq_min=0.0, acq_max=1.0,
                 feas_approach='fwa', feas_param=1.0):
        self.bayesian_network = bayesian_network
        self.sampling_param = sampling_param
        self.acq_min = acq_min
        self.acq_max = acq_max
        self.inv_range = 1.0 / max(acq_max - acq_min, 1e-8)

        self.frac_infeasible = bayesian_network.prior_1
        self.feas_approach = feas_approach
        self.feas_param = feas_param

        if self.frac_infeasible < 1e-6:
            self.acquisition_function = self._acquisition_all_feasible
        elif 1.0 - self.frac_infeasible < 1e-6:
            self.acquisition_function = self._acquisition_all_infeasible
        else:
            if feas_approach == 'fwa':
                self.acquisition_function = self._fwa_acquisition
            elif feas_approach == 'fia':
                self.acquisition_function = self._fia_acquisition
                self.feasibility_weight = (self.frac_infeasible) ** feas_param
            elif feas_approach == 'fca':
                self.acquisition_function = self._acquisition_all_feasible

    def __call__(self, x):
        return self.acquisition_function(x)

    def _fwa_acquisition(self, x):
        num, inv_den, _ = self.bayesian_network.kernel_contribution(x)
        prob_feas = self.bayesian_network.prob_feasible(x)
        acq_samp = (num + self.sampling_param) * inv_den
        acq_norm = (acq_samp - self.acq_min) * self.inv_range
        acq_maximize = 1.0 - acq_norm
        return -(acq_maximize * prob_feas)

    def _fia_acquisition(self, x):
        num, inv_den, _ = self.bayesian_network.kernel_contribution(x)
        prob_infeas = self.bayesian_network.prob_infeasible(x)
        acq_samp = (num + self.sampling_param) * inv_den
        acq_norm = (acq_samp - self.acq_min) * self.inv_range
        return self.feasibility_weight * prob_infeas + (1.0 - self.feasibility_weight) * acq_norm

    def _acquisition_all_feasible(self, x):
        num, inv_den, _ = self.bayesian_network.kernel_contribution(x)
        acq_samp = (num + self.sampling_param) * inv_den
        acq_norm = (acq_samp - self.acq_min) * self.inv_range
        return acq_norm

    def _acquisition_all_infeasible(self, x):
        return self.bayesian_network.infeasible_kernel_density(x)


class Acquisition:
    """
    Orchestrates acquisition function evaluation and candidate optimization.
    """
    def __init__(self, config, known_constraints=None):
        self.config = config
        self.known_constraints = known_constraints
        self.feas_approach = config.feas_approach
        self.feas_param = config.feas_param
        self.acquisition_functions = {}

    def _feasibility_constraint(self, param_dict):
        x = np.array([param_dict[name] for name in self.config.param_names], dtype=np.float64)
        return self.bayesian_network.classification_surrogate(x, threshold=self.feas_param)

    def propose(self, best_params, bayesian_network, sampling_param_values, num_samples_per_dim=200):
        self.bayesian_network = bayesian_network
        self.sampling_param_values = sampling_param_values
        self.acquisition_functions = {}

        constraints = []
        if self.known_constraints:
            if isinstance(self.known_constraints, list):
                constraints.extend(self.known_constraints)
            else:
                constraints.append(self.known_constraints)

        if self.feas_approach == 'fca' and self.bayesian_network.prior_1 > 1e-6 and self.bayesian_network.prior_0 > 1e-6:
            constraints.append(self._feasibility_constraint)

        # Draw random proposals (uniform + perturbed around incumbent)
        random_sampler = RandomSampler(self.config, constraints=constraints)
        num_samples = num_samples_per_dim * self.config.num_features
        
        uniform_samples = random_sampler.draw(num=num_samples)
        perturbed_samples = random_sampler.perturb(best_params, num=num_samples)
        random_proposals = np.concatenate([uniform_samples, perturbed_samples], axis=0)

        # Optimize proposals for each sampling strategy λ
        optimizer = GradientOptimizer(self.config, constraints=constraints)
        all_proposals = []

        for batch_index, sampling_param in enumerate(sampling_param_values):
            # Estimate approximate min and max of acquisition function for scaling
            acq_raw = []
            for prop in random_proposals[:50]:
                num, inv_den, _ = self.bayesian_network.kernel_contribution(prop)
                acq_raw.append((num + sampling_param) * inv_den)
            acq_min = np.min(acq_raw)
            acq_max = np.max(acq_raw)

            acq_func = AcquisitionFunction(
                bayesian_network=self.bayesian_network,
                sampling_param=sampling_param,
                acq_min=acq_min,
                acq_max=acq_max,
                feas_approach=self.feas_approach,
                feas_param=self.feas_param
            )
            self.acquisition_functions[batch_index] = acq_func

            # Gradient optimization starting from top 20% random proposals
            eval_vals = np.array([acq_func(p) for p in random_proposals])
            top_indices = np.argsort(eval_vals)[:max(int(0.2 * len(random_proposals)), 10)]
            seed_proposals = random_proposals[top_indices]

            optimized = optimizer.optimize(acq_func, seed_proposals, max_iter=15)
            combined = np.concatenate([random_proposals, optimized], axis=0)
            all_proposals.append(combined)

        return np.array(all_proposals, dtype=np.float64)

    def eval_acquisition(self, x, batch_index):
        return self.acquisition_functions[batch_index](x)
