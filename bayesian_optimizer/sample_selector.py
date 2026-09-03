import numpy as np

class SampleSelector:
    """
    Diversity-Aware Sample Selector.
    Selects parameter recommendations by penalizing proposals that are close
    to previously observed points or previously selected recommendations in the batch.
    """
    def __init__(self, config):
        self.config = config
        self.dist_param = getattr(config, 'dist_param', 0.5)

    def select(self, num_batches, proposals, eval_acquisition, sampling_param_values, obs_params):
        """
        num_batches: int, number of batches requested
        proposals: (num_sampling_strategies, num_proposals, num_features)
        eval_acquisition: callable (x, batch_index) -> acq_value
        sampling_param_values: 1D array of sampling strategy λ values
        obs_params: (num_obs, num_features) array of past observations
        """
        num_strategies = len(sampling_param_values)
        num_props = proposals.shape[1]

        # Compute exp(-acq) reward matrix of shape (num_strategies, num_props)
        exp_objs = np.zeros((num_strategies, num_props), dtype=np.float64)
        for s_idx in range(num_strategies):
            for p_idx in range(num_props):
                acq_val = eval_acquisition(proposals[s_idx, p_idx], s_idx)
                exp_objs[s_idx, p_idx] = np.exp(-acq_val)

        # Normalize parameter bounds to [0, 1] for distance computations
        lowers = self.config.param_lowers
        uppers = self.config.param_uppers
        ranges = uppers - lowers
        ranges = np.where(ranges < 1e-8, 1.0, ranges)

        obs_params_norm = (obs_params - lowers) / ranges
        proposals_norm = (proposals - lowers) / ranges

        # Zero out reward for proposals that duplicate existing observations
        for s_idx in range(num_strategies):
            batch_props_norm = proposals_norm[s_idx]
            for p_idx in range(num_props):
                if len(obs_params_norm) > 0:
                    dists = np.sqrt(np.sum((obs_params_norm - batch_props_norm[p_idx])**2, axis=1))
                    if np.min(dists) < 1e-5:
                        exp_objs[s_idx, p_idx] = 0.0

        # Characteristic distance penalty scale based on observation density
        num_obs = max(len(obs_params), 1)
        char_dists = ranges / float(num_obs)**self.dist_param

        selected_samples = []

        for b_idx in range(num_batches):
            for s_idx in range(num_strategies):
                batch_props = proposals[s_idx]
                div_crits = np.ones(num_props, dtype=np.float64)

                for p_idx in range(num_props):
                    prop = batch_props[p_idx]
                    
                    # Compute min distance to observed samples
                    if len(obs_params) > 0:
                        obs_dists = np.min([np.abs(prop - x) for x in obs_params], axis=0)
                    else:
                        obs_dists = ranges

                    # Compute min distance to newly selected batch samples
                    if len(selected_samples) > 0:
                        sel_dists = np.min([np.abs(prop - x) for x in selected_samples], axis=0)
                        min_dists = np.minimum(obs_dists, sel_dists)
                    else:
                        min_dists = obs_dists

                    div_crits[p_idx] = np.minimum(1.0, np.mean(np.exp(2.0 * (min_dists - char_dists) / ranges)))

                # Reweight rewards by diversity multiplier
                reweighted = exp_objs[s_idx] * div_crits
                best_idx = np.argmax(reweighted)

                selected = batch_props[best_idx].copy()
                selected_samples.append(selected)

                # Set reward of selected proposal to 0 to prevent re-selection
                exp_objs[s_idx, best_idx] = 0.0

        return np.array(selected_samples, dtype=np.float64)
