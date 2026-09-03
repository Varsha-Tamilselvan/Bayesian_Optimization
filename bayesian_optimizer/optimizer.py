import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np

from .config import Config
from .observation_processor import ObservationProcessor, param_vectors_to_dicts, param_dicts_to_vectors
from .random_sampler import RandomSampler
from .bayesian_network import BayesianNetwork
from .acquisition import Acquisition
from .sample_selector import SampleSelector

class BayesianOptimizer:
    """
    Gryffin-Inspired Multi-Objective Bayesian Optimizer for Continuous Parameters.
    """
    def __init__(self, config_dict=None, known_constraints=None):
        self.config = Config(config_dict)
        self.known_constraints = known_constraints

        self.random_sampler = RandomSampler(self.config, constraints=self.known_constraints)
        self.obs_processor = ObservationProcessor(self.config)
        self.bayesian_network = BayesianNetwork(self.config, frac_feas=1.0)
        self.acquisition = Acquisition(self.config, known_constraints=self.known_constraints)
        self.sample_selector = SampleSelector(self.config)

        self.iter_counter = 0

    def recommend(self, observations=None, sampling_strategies=None, num_batches=None, as_array=False):
        """
        Recommends the next set of experimental parameter features based on observations.
        
        Args:
            observations: List of dicts, e.g. [{'temp': 50.0, 'ph': 7.0, 'target1': 0.8, 'target2': 1.2}, ...]
            sampling_strategies: List of float values or None (default linspace 1.0 to -1.0)
            num_batches: int or None (default config.batches)
            as_array: bool, whether to return numpy array or list of dicts

        Returns:
            List of dicts or NumPy array with recommended experimental feature values.
        """
        if sampling_strategies is None:
            num_strat = self.config.sampling_strategies
            sampling_strategies = np.linspace(1.0, -1.0, num_strat)
        else:
            sampling_strategies = np.array(sampling_strategies, dtype=np.float64)

        batches = num_batches if num_batches is not None else self.config.batches

        # 1. Fallback to random sampling if no observations yet
        if observations is None or len(observations) == 0:
            num_samples = batches * len(sampling_strategies)
            samples = self.random_sampler.draw(num=num_samples)
            if as_array:
                return samples
            return param_vectors_to_dicts(samples, self.config.param_names)

        # 2. Process observations
        obs_params, obs_objs, obs_feas, mask_kwn, mask_mirror = self.obs_processor.process_observations(observations)

        # 3. Compute sampling parameter λ values (exploitation vs exploration)
        sampling_param_values = sampling_strategies * self.bayesian_network.inverse_volume

        # 4. Train BNN & build kernel density surrogates
        self.bayesian_network.sample(obs_params)
        self.bayesian_network.build_kernels(obs_objs=obs_objs, obs_feas=obs_feas, mask_kwn=mask_kwn)

        # 5. Identify incumbent (best parameter set)
        if np.sum(mask_kwn) > 0:
            best_idx = np.argmin(obs_objs[mask_kwn])
            best_params = obs_params[mask_kwn][best_idx]
        else:
            best_params = obs_params[np.random.randint(0, len(obs_params))]

        # 6. Optimize acquisition function to generate proposals
        proposals = self.acquisition.propose(
            best_params=best_params,
            bayesian_network=self.bayesian_network,
            sampling_param_values=sampling_param_values,
            num_samples_per_dim=self.config.num_random_samples
        )

        # 7. Select diverse recommendations
        samples = self.sample_selector.select(
            num_batches=batches,
            proposals=proposals,
            eval_acquisition=self.acquisition.eval_acquisition,
            sampling_param_values=sampling_param_values,
            obs_params=obs_params
        )

        self.iter_counter += 1

        if as_array:
            return samples
        return param_vectors_to_dicts(samples, self.config.param_names)

    def get_regression_surrogate(self, params):
        """
        Evaluate objective surrogate model predictions for input parameter points.
        params: List of dicts or NumPy array (N, D)
        """
        if isinstance(params, list):
            X = param_dicts_to_vectors(params, self.config.param_names)
        else:
            X = np.array(params, dtype=np.float64)

        preds = [self.bayesian_network.regression_surrogate(x) for x in X]
        return np.array(preds)

    def get_feasibility_surrogate(self, params, threshold=0.5):
        """
        Evaluate feasibility probability for input parameter points.
        params: List of dicts or NumPy array (N, D)
        """
        if isinstance(params, list):
            X = param_dicts_to_vectors(params, self.config.param_names)
        else:
            X = np.array(params, dtype=np.float64)

        preds = [self.bayesian_network.prob_feasible(x) for x in X]
        return np.array(preds)
