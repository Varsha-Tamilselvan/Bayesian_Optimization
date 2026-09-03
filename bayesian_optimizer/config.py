import numpy as np

class Config:
    """
    Configuration Parser for Bayesian Optimizer (Gryffin-inspired).
    Tailored for continuous parameters and multi-objective optimization with constraint support.
    """
    def __init__(self, config_dict=None):
        self.config_dict = config_dict or {}

        # Parse general settings
        gen = self.config_dict.get('general', {})
        self.sampling_strategies = gen.get('sampling_strategies', 2)
        self.batches = gen.get('batches', 1)
        self.num_cpus = gen.get('num_cpus', 1)
        self.dist_param = gen.get('dist_param', 0.5)
        self.feas_approach = gen.get('feas_approach', 'fwa')  # 'fwa', 'fia', or 'fca'
        self.feas_param = gen.get('feas_param', 1.0)
        self.random_seed = gen.get('random_seed', None)
        self.obj_transform = gen.get('obj_transform', 'sqrt')  # None, 'sqrt', 'cbrt', 'square'
        self.num_random_samples = gen.get('num_random_samples', 200)
        self.reject_tol = gen.get('reject_tol', 1000)
        self.boosted = gen.get('boosted', True)

        # Parse BNN model configuration
        model = self.config_dict.get('model', {})
        self.num_epochs = model.get('num_epochs', 1500)
        self.learning_rate = model.get('learning_rate', 0.05)
        self.num_draws = model.get('num_draws', 500)
        self.hidden_shape = model.get('hidden_shape', 12)
        self.num_layers = model.get('num_layers', 3)

        # Parse parameters (Features)
        self.parameters = self.config_dict.get('parameters', [])
        self.param_names = [p['name'] for p in self.parameters]
        self.param_lowers = np.array([p['low'] for p in self.parameters], dtype=np.float64)
        self.param_uppers = np.array([p['high'] for p in self.parameters], dtype=np.float64)
        self.param_ranges = self.param_uppers - self.param_lowers
        self.param_periodic = [p.get('periodic', False) for p in self.parameters]
        self.num_features = len(self.parameters)

        # Since parameters are continuous, kernel mappings are 1-to-1
        self.kernel_names = list(self.param_names)
        self.kernel_lowers = np.copy(self.param_lowers)
        self.kernel_uppers = np.copy(self.param_uppers)
        self.kernel_ranges = np.copy(self.param_ranges)
        self.kernel_sizes = np.ones(self.num_features, dtype=int)
        self.kernel_types = ['continuous'] * self.num_features
        self.feature_types = ['continuous'] * self.num_features
        self.feature_lowers = np.copy(self.param_lowers)
        self.feature_uppers = np.copy(self.param_uppers)
        self.feature_ranges = np.copy(self.param_ranges)
        self.feature_sizes = np.ones(self.num_features, dtype=int)
        self.feature_process_constrained = np.zeros(self.num_features, dtype=bool)
        self.process_constrained = False

        # Parse objectives (Target columns)
        self.objectives = self.config_dict.get('objectives', [])
        self.obj_names = [o['name'] for o in self.objectives]
        self.obj_goals = [o.get('goal', 'min') for o in self.objectives]
        self.obj_tolerances = [o.get('tolerance', 0.0) for o in self.objectives]
        self.obj_absolutes = [o.get('absolute', False) for o in self.objectives]
        self.num_objectives = len(self.objectives)

    def get(self, key, default=None):
        return getattr(self, key, default)
