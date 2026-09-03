import numpy as np

class KernelEvaluator:
    """
    Pure NumPy Vectorized Implementation of Gaussian Kernel Density Evaluations.
    Replaces Cython kernel_evaluations.pyx without binary compilation dependencies.
    """
    def __init__(self, locs, sqrt_precs, cat_probs, kernel_types, kernel_sizes, kernel_ranges,
                 lower_prob_bound, objs, inv_vol):
        self.locs = locs                    # Shape: (num_samples, num_obs, num_features)
        self.sqrt_precs = sqrt_precs        # Shape: (num_samples, num_obs, num_features)
        self.cat_probs = cat_probs
        self.kernel_types = kernel_types
        self.kernel_sizes = kernel_sizes
        self.kernel_ranges = kernel_ranges
        self.lower_prob_bound = lower_prob_bound
        self.objs = objs                    # Shape: (num_obs,)
        self.inv_vol = inv_vol

        self.num_samples, self.num_obs, self.num_features = locs.shape
        self.inv_sqrt_two_pi = 0.3989422804014327

    def _probs(self, sample):
        """
        Computes kernel probabilities p(sample | obs_i) averaged over BNN samples.
        sample: 1D array of shape (num_features,)
        Returns: 1D array of shape (num_obs,)
        """
        # Vectorized over samples and obs:
        # diff: (num_samples, num_obs, num_features)
        diff = sample.reshape(1, 1, -1) - self.locs
        
        # Continuous non-periodic Gaussian exponent:
        # 0.5 * (sqrt_prec * diff)^2
        arg = 0.5 * (self.sqrt_precs * diff)**2
        
        # Product of precision terms: (num_samples, num_obs)
        prec_prod = np.prod(self.sqrt_precs, axis=2)
        
        # Sum exponent over features: (num_samples, num_obs)
        exp_sum = np.sum(arg, axis=2)
        
        # Mask out extreme exponent values to avoid underflow
        exp_term = np.where(exp_sum > 200.0, 0.0, np.exp(-exp_sum))
        
        # Combine precision product, constant, and exponential
        sample_obs_probs = prec_prod * (self.inv_sqrt_two_pi ** self.num_features) * exp_term
        
        # Average probability across BNN posterior draws
        probs = np.mean(sample_obs_probs, axis=0)
        return probs

    def get_kernel_contrib(self, sample):
        """
        Computes numerator and inverse denominator for acquisition function.
        """
        probs_sample = self._probs(sample)
        num = np.sum(self.objs * probs_sample)
        den = np.sum(probs_sample)
        inv_den = 1.0 / (self.inv_vol + den)
        return num, inv_den, probs_sample

    def get_regression_surrogate(self, sample):
        """
        Evaluates Nadaraya-Watson kernel regression surrogate model prediction at sample.
        """
        probs_sample = self._probs(sample)
        num = np.sum(self.objs * probs_sample)
        den = np.sum(probs_sample)
        return num / (den + 1e-8)

    def get_binary_kernel_densities(self, sample):
        """
        Computes log kernel densities for feasible (0) and infeasible (1) observations.
        """
        probs_sample = self._probs(sample)
        
        mask_feas = (self.objs <= 0.5)
        mask_infeas = (self.objs > 0.5)

        density_0 = np.sum(probs_sample[mask_feas])
        num_0 = np.sum(mask_feas)

        density_1 = np.sum(probs_sample[mask_infeas])
        num_1 = np.sum(mask_infeas)

        log_density_0 = np.log(max(density_0, 1e-300)) - np.log(max(num_0, 1.0))
        log_density_1 = np.log(max(density_1, 1e-300)) - np.log(max(num_1, 1.0))

        return log_density_0, log_density_1

    def get_probability_of_feasibility(self, sample, log_prior_0, log_prior_1):
        """
        Posterior probability P(feasible | sample).
        """
        log_density_0, log_density_1 = self.get_binary_kernel_densities(sample)
        
        post_0 = np.exp(np.clip(log_density_0 + log_prior_0, -300, 300))
        post_1 = np.exp(np.clip(log_density_1 + log_prior_1, -300, 300))

        total_post = post_0 + post_1
        if total_post < 1e-100 or np.isnan(total_post):
            p0 = np.exp(log_prior_0) if np.isfinite(log_prior_0) else 1.0
            p1 = np.exp(log_prior_1) if np.isfinite(log_prior_1) else 0.0
            return p0 / (p0 + p1 + 1e-10)

        return post_0 / total_post

    def get_probability_of_infeasibility(self, sample, log_prior_0, log_prior_1):
        """
        Posterior probability P(infeasible | sample).
        """
        return 1.0 - self.get_probability_of_feasibility(sample, log_prior_0, log_prior_1)
