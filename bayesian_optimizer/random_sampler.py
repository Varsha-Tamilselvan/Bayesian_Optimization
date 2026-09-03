import numpy as np
from .observation_processor import param_vector_to_dict

class RandomSampler:
    """
    Random Sampler with Support for Known User Constraints.
    """
    def __init__(self, config, constraints=None):
        self.config = config
        self.reject_tol = getattr(config, 'reject_tol', 1000)

        if constraints is not None and not isinstance(constraints, list):
            self.constraints = [constraints]
        else:
            self.constraints = constraints

    def draw(self, num=1):
        if self.constraints is None or len(self.constraints) == 0:
            return self._fast_draw(num=num)
        else:
            return self._slow_draw(num=num)

    def perturb(self, ref_sample, num=1, scale=0.05):
        if self.constraints is None or len(self.constraints) == 0:
            return self._fast_perturb(ref_sample, num=num, scale=scale)
        else:
            return self._slow_perturb(ref_sample, num=num, scale=scale)

    def _fast_draw(self, num=1):
        lowers = self.config.param_lowers
        uppers = self.config.param_uppers
        samples = np.random.uniform(low=lowers, high=uppers, size=(num, len(lowers)))
        return samples.astype(np.float64)

    def _slow_draw(self, num=1):
        samples = []
        counter = 0
        lowers = self.config.param_lowers
        uppers = self.config.param_uppers
        num_features = len(lowers)

        while len(samples) < num:
            sample = np.random.uniform(low=lowers, high=uppers, size=num_features)
            param_dict = param_vector_to_dict(sample, self.config.param_names)

            is_feasible = True
            for constr in self.constraints:
                try:
                    res = constr(param_dict)
                except Exception:
                    res = constr(sample)
                if not res:
                    is_feasible = False
                    break

            if is_feasible:
                samples.append(sample)

            counter += 1
            if counter > self.reject_tol * max(num, 10):
                # Fallback to fast draw if constraints are too restrictive or reject_tol exceeded
                if len(samples) == 0:
                    return self._fast_draw(num=num)
                while len(samples) < num:
                    samples.append(samples[len(samples) % len(samples)])
                break

        return np.array(samples, dtype=np.float64)

    def _fast_perturb(self, ref_sample, num=1, scale=0.05):
        lowers = self.config.param_lowers
        uppers = self.config.param_uppers
        ranges = self.config.param_ranges
        
        noise = np.random.uniform(low=-scale, high=scale, size=(num, len(ref_sample))) * ranges
        perturbed = ref_sample + noise
        perturbed = np.clip(perturbed, lowers, uppers)
        return perturbed.astype(np.float64)

    def _slow_perturb(self, ref_sample, num=1, scale=0.05):
        samples = []
        counter = 0
        new_scale = scale

        while len(samples) < num:
            perturbed = self._fast_perturb(ref_sample, num=1, scale=new_scale)[0]
            param_dict = param_vector_to_dict(perturbed, self.config.param_names)

            is_feasible = True
            for constr in self.constraints:
                try:
                    res = constr(param_dict)
                except Exception:
                    res = constr(perturbed)
                if not res:
                    is_feasible = False
                    break

            if is_feasible:
                samples.append(perturbed)

            counter += 1
            if counter > 100 * (len(samples) + 1):
                new_scale *= 1.5
            if counter > self.reject_tol * max(num, 10):
                if len(samples) == 0:
                    samples.append(ref_sample)
                while len(samples) < num:
                    samples.append(samples[0])
                break

        return np.array(samples, dtype=np.float64)
