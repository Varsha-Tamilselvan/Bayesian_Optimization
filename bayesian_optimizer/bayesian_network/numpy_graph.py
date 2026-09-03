import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

class NumpyGraph:
    """
    NumPy-based forward pass graph to compute kernel density parameters
    (locations and square-root precisions) from BNN posterior draws.
    """
    def __init__(self, config):
        self.config = config

    def declare_training_data(self, features):
        self.num_obs = len(features)
        self.features = features.detach().numpy() if hasattr(features, 'detach') else features

    def compute_kernels(self, posteriors, frac_feas, num_draws=500):
        frac_feas = max(frac_feas, 1e-4)
        tau_rescaling = np.tile(self.config.kernel_ranges**2, (self.num_obs, 1))

        # Forward pass through MLP layers for each posterior weight sample
        post_layer_outputs = [np.array([self.features for _ in range(num_draws)])]
        num_layers = self.config.num_layers

        for layer_index in range(num_layers):
            weight = posteriors[f'weight_{layer_index}']
            bias = posteriors[f'bias_{layer_index}']

            # Activation: ReLU for hidden layers, linear for final layer
            activation = (lambda x: np.maximum(x, 0.0)) if layer_index < num_layers - 1 else (lambda x: x)

            outputs = []
            for s_idx in range(len(weight)):
                w = weight[s_idx]
                b = bias[s_idx]
                out = activation(np.matmul(post_layer_outputs[-1][s_idx], w) + b)
                outputs.append(out)

            post_layer_outputs.append(np.array(outputs))

        post_bnn_output = post_layer_outputs[-1]

        # Draw precision parameters tau from Gamma distribution
        shape_param = 12.0 * (self.num_obs / frac_feas)**2
        post_tau_normed = np.random.gamma(shape_param + np.zeros_like(post_bnn_output), np.ones_like(post_bnn_output))
        post_tau = post_tau_normed / (tau_rescaling + 1e-8)
        post_sqrt_tau = np.sqrt(post_tau)
        post_scale = 1.0 / (post_sqrt_tau + 1e-8)

        # Map predictions to search domain
        post_kernels = {}
        lowers = self.config.kernel_lowers
        uppers = self.config.kernel_uppers

        for target_idx in range(self.config.num_features):
            post_relevant = post_bnn_output[:, :, target_idx:target_idx+1]
            low = lowers[target_idx]
            up = uppers[target_idx]

            post_support = (up - low) * (1.2 * sigmoid(post_relevant) - 0.1) + low
            post_kernels[f'param_{target_idx}'] = {
                'loc': post_support,
                'sqrt_prec': post_sqrt_tau[:, :, target_idx:target_idx+1],
                'scale': post_scale[:, :, target_idx:target_idx+1]
            }

        return post_kernels
