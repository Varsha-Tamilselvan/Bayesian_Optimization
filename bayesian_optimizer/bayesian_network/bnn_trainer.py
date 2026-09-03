import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
import numpy as np
from .numpy_graph import NumpyGraph

class BayesLinear(nn.Module):
    """
    Bayesian Linear Layer with Gaussian priors N(0, 1) and variational posterior N(mu, exp(log_sigma)).
    Replaces torchbnn dependency with standard PyTorch.
    """
    def __init__(self, in_features, out_features, prior_mu=0.0, prior_sigma=1.0):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-3.0))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features).fill_(-3.0))

        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

    def forward(self, x):
        if self.training:
            weight_std = torch.exp(self.weight_log_sigma)
            weight_eps = torch.randn_like(self.weight_mu)
            weight = self.weight_mu + weight_std * weight_eps

            bias_std = torch.exp(self.bias_log_sigma)
            bias_eps = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + bias_std * bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def kl_loss(self):
        weight_std = torch.exp(self.weight_log_sigma)
        kl_w = torch.sum(self.weight_log_sigma - torch.log(torch.tensor(self.prior_sigma)) + 
                         (self.prior_sigma**2 + (self.weight_mu - self.prior_mu)**2) / (2 * weight_std**2) - 0.5)

        bias_std = torch.exp(self.bias_log_sigma)
        kl_b = torch.sum(self.bias_log_sigma - torch.log(torch.tensor(self.prior_sigma)) + 
                         (self.prior_sigma**2 + (self.bias_mu - self.prior_mu)**2) / (2 * bias_std**2) - 0.5)
        return kl_w + kl_b


class BNN(nn.Module):
    def __init__(self, config, num_observations, frac_feas):
        super(BNN, self).__init__()
        self.config = config
        self.feature_size = config.num_features
        self.bnn_output_size = config.num_features
        self.num_obs = num_observations
        self.frac_feas = max(frac_feas, 1e-4)

        self.hidden_shape = config.hidden_shape
        self.num_layers = config.num_layers
        self.num_draws = config.num_draws

        self.numpy_graph = NumpyGraph(config)

        # Build Bayesian MLP layers
        layers = []
        layers.append(BayesLinear(self.feature_size, self.hidden_shape))
        layers.append(nn.ReLU())

        for _ in range(2, self.num_layers):
            layers.append(BayesLinear(self.hidden_shape, self.hidden_shape))
            layers.append(nn.ReLU())

        layers.append(BayesLinear(self.hidden_shape, self.bnn_output_size))
        self.layers = nn.Sequential(*layers)

        # Precision parameters tau ~ Gamma(concentration, rate)
        self.tau_rescaling = torch.tensor(config.kernel_ranges**2, dtype=torch.float32).unsqueeze(0).repeat(self.num_obs, 1)
        init_conc = 12.0 * (self.num_obs / self.frac_feas)**2
        self.gamma_concentration = nn.Parameter(torch.zeros(self.num_obs, self.bnn_output_size) + init_conc)
        self.gamma_rate = nn.Parameter(F.softplus(torch.ones(self.num_obs, self.bnn_output_size)))

    def forward(self, x, y):
        out = self.layers(x)
        conc = F.softplus(self.gamma_concentration, threshold=0.01)
        rate = F.softplus(self.gamma_rate, threshold=0.01)
        tau_normed = td.gamma.Gamma(conc, rate)

        scale = 1.0 / torch.sqrt(tau_normed.rsample() / self.tau_rescaling + 1e-8)

        # Map predictions to feature bounds
        lowers = torch.tensor(self.config.kernel_lowers, dtype=torch.float32)
        uppers = torch.tensor(self.config.kernel_uppers, dtype=torch.float32)
        post_support = (uppers - lowers) * (1.2 * torch.sigmoid(out) - 0.1) + lowers

        post_predict = td.normal.Normal(post_support, scale)
        return [{'pred': post_predict, 'target': y}]

    def kl_loss(self):
        loss = 0.0
        for module in self.layers:
            if isinstance(module, BayesLinear):
                loss = loss + module.kl_loss()
        return loss

    def register_numpy_graph(self, features):
        self.numpy_graph.declare_training_data(features)

    def _sample(self, num_draws):
        posterior_samples = {}
        idx = 0
        for module in self.layers:
            if isinstance(module, BayesLinear):
                w_dist = td.normal.Normal(module.weight_mu, torch.exp(module.weight_log_sigma))
                b_dist = td.normal.Normal(module.bias_mu, torch.exp(module.bias_log_sigma))

                w_sample = w_dist.sample(sample_shape=(num_draws,)).transpose(-2, -1)
                b_sample = b_dist.sample(sample_shape=(num_draws,))

                posterior_samples[f'weight_{idx}'] = w_sample.detach().numpy()
                posterior_samples[f'bias_{idx}'] = b_sample.detach().numpy()
                idx += 1

        post_kernels = self.numpy_graph.compute_kernels(posterior_samples, self.frac_feas, num_draws=num_draws)
        self.trace = post_kernels

    def get_kernels(self, num_draws=None):
        num_draws = num_draws or self.num_draws
        self._sample(num_draws)

        trace_kernels = {'locs': [], 'sqrt_precs': [], 'probs': []}
        for param_index in range(len(self.config.param_names)):
            post_kernel = self.trace[f'param_{param_index}']
            trace_kernels['locs'].append(post_kernel['loc'].astype(np.float64))
            trace_kernels['sqrt_precs'].append(post_kernel['sqrt_prec'].astype(np.float64))
            trace_kernels['probs'].append(np.zeros_like(post_kernel['loc'], dtype=np.float64))

        for key in trace_kernels:
            trace_kernels[key] = np.concatenate(trace_kernels[key], axis=2)

        return trace_kernels


class BNNTrainer:
    def __init__(self, config, frac_feas=1.0):
        self.config = config
        self.frac_feas = frac_feas

    def train(self, observed_params):
        if self.config.random_seed is not None:
            torch.manual_seed(self.config.random_seed)

        obs_tensor = torch.tensor(observed_params, dtype=torch.float32)
        features, targets = self._generate_train_data(obs_tensor)
        num_obs = len(observed_params)

        model = BNN(self.config, num_obs, self.frac_feas)
        model.register_numpy_graph(features)

        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

        for _ in range(self.config.num_epochs):
            inferences = model(features, targets)
            nll = 0.0
            for inf in inferences:
                nll = nll - torch.sum(inf['pred'].log_prob(inf['target']))

            kl = model.kl_loss() / num_obs
            loss = nll + kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model

    def _generate_train_data(self, observed_params):
        lowers = torch.tensor(self.config.kernel_lowers, dtype=torch.float32)
        uppers = torch.tensor(self.config.kernel_uppers, dtype=torch.float32)
        ranges = uppers - lowers
        ranges = torch.where(ranges < 1e-8, torch.ones_like(ranges), ranges)

        rescaled_features = (observed_params - lowers) / ranges
        rescaled_targets = rescaled_features.clone()
        return rescaled_features, rescaled_targets
