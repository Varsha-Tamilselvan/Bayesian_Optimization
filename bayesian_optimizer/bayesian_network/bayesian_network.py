import numpy as np

from .bnn_trainer import BNNTrainer
from .kernel_evaluator import KernelEvaluator

class BayesianNetwork:
    """
    Bayesian Network module. Trains the BNN and builds regression/classification kernels.
    """
    def __init__(self, config, frac_feas=1.0):
        self.config = config
        self.frac_feas = frac_feas

        # Search domain volume
        self.volume = np.prod(config.param_ranges)
        if self.volume <= 0:
            self.volume = 1.0
        self.inverse_volume = 1.0 / self.volume

        self.trace_kernels = None
        self.kernel_regression = None
        self.kernel_classification = None

        self.prior_0 = 1.0
        self.prior_1 = 0.0
        self.log_prior_0 = 0.0
        self.log_prior_1 = -np.inf

        self.lower_prob_bound = 1e-25

    def sample(self, obs_params):
        """Train BNN and extract kernel parameters (locations and precisions)"""
        trainer = BNNTrainer(self.config, self.frac_feas)
        model = trainer.train(obs_params)
        self.trace_kernels = model.get_kernels()

    def build_kernels(self, obs_objs, obs_feas, mask_kwn):
        """Build kernel density evaluators for regression and feasibility classification"""
        assert self.trace_kernels is not None, "Must run sample() before building kernels."

        self.obs_objs_kwn = obs_objs[mask_kwn]
        self.obs_objs_feas = obs_feas

        # Compute feasible (0) and infeasible (1) priors
        n_tot = len(self.obs_objs_feas)
        if n_tot > 0:
            self.prior_0 = np.sum(self.obs_objs_feas < 0.5) / n_tot
            self.prior_1 = np.sum(self.obs_objs_feas >= 0.5) / n_tot
        else:
            self.prior_0 = 1.0
            self.prior_1 = 0.0

        self.log_prior_0 = np.log(self.prior_0) if self.prior_0 > 0.0 else -np.inf
        self.log_prior_1 = np.log(self.prior_1) if self.prior_1 > 0.0 else -np.inf

        locs_all = self.trace_kernels['locs']
        sqrt_precs_all = self.trace_kernels['sqrt_precs']
        probs_all = self.trace_kernels['probs']

        locs_kwn = locs_all[:, mask_kwn, :]
        sqrt_precs_kwn = sqrt_precs_all[:, mask_kwn, :]
        probs_kwn = probs_all[:, mask_kwn, :]

        kernel_types = np.zeros(self.config.num_features, dtype=int)
        kernel_sizes = self.config.kernel_sizes
        kernel_ranges = self.config.kernel_ranges

        # Kernel regression model (on feasible/known points)
        self.kernel_regression = KernelEvaluator(
            locs=locs_kwn, sqrt_precs=sqrt_precs_kwn, cat_probs=probs_kwn,
            kernel_types=kernel_types, kernel_sizes=kernel_sizes, kernel_ranges=kernel_ranges,
            lower_prob_bound=self.lower_prob_bound, objs=self.obs_objs_kwn, inv_vol=self.inverse_volume
        )

        # Kernel classification model (on all observations including infeasible points)
        self.kernel_classification = KernelEvaluator(
            locs=locs_all, sqrt_precs=sqrt_precs_all, cat_probs=probs_all,
            kernel_types=kernel_types, kernel_sizes=kernel_sizes, kernel_ranges=kernel_ranges,
            lower_prob_bound=self.lower_prob_bound, objs=self.obs_objs_feas, inv_vol=self.inverse_volume
        )

    def kernel_contribution(self, sample):
        return self.kernel_regression.get_kernel_contrib(sample)

    def regression_surrogate(self, sample):
        return self.kernel_regression.get_regression_surrogate(sample)

    def prob_feasible(self, sample):
        return self.kernel_classification.get_probability_of_feasibility(sample, self.log_prior_0, self.log_prior_1)

    def prob_infeasible(self, sample):
        return self.kernel_classification.get_probability_of_infeasibility(sample, self.log_prior_0, self.log_prior_1)

    def classification_surrogate(self, sample, threshold=0.5):
        return self.prob_feasible(sample) >= threshold

    def infeasible_kernel_density(self, sample):
        _, log_density_1 = self.kernel_classification.get_binary_kernel_densities(sample)
        return np.exp(log_density_1)
