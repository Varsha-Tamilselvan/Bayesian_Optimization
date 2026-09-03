import numpy as np

class MultiObjectiveScalarizer:
    """
    Hierarchical multi-objective scalarization (inspired by Chimera).
    Scalarizes multiple target columns into a single merit value in [0, 1].
    Supports objective goals ('min' or 'max'), tolerances, and absolute flags.
    """
    def __init__(self, goals, tolerances, absolutes, softness=0.001):
        self.goals = goals
        self.tolerances = tolerances
        self.absolutes = absolutes
        self.softness = softness

    def scalarize(self, raw_objs):
        """
        raw_objs: np.ndarray of shape (N, num_objs)
        Returns: np.ndarray of shape (N,) scaled between 0 (best) and 1 (worst)
        """
        num_obs, num_objs = raw_objs.shape
        if num_obs == 0:
            return np.array([])
        if num_obs == 1:
            return np.zeros(1)

        # Step 1: Normalize each objective based on goal (0 = best, 1 = worst)
        norm_objs = np.zeros_like(raw_objs, dtype=np.float64)
        for i in range(num_objs):
            col = raw_objs[:, i]
            col_min, col_max = np.min(col), np.max(col)
            if col_max - col_min > 1e-8:
                scaled = (col - col_min) / (col_max - col_min)
            else:
                scaled = np.zeros_like(col)

            if self.goals[i] in ['max', 'maximize']:
                norm_objs[:, i] = 1.0 - scaled
            else:
                norm_objs[:, i] = scaled

        if num_objs == 1:
            return norm_objs[:, 0]

        # Step 2: Combine objectives hierarchically with smooth thresholding
        merits = norm_objs[:, -1].copy()
        for i in reversed(range(num_objs - 1)):
            obj_i = norm_objs[:, i]
            tol_i = self.tolerances[i]
            # Smooth step function using arctan or sigmoid for thresholding
            shift = obj_i - tol_i
            weight = 0.5 * (1.0 + np.tanh(shift / (self.softness + 1e-5)))
            merits = weight * obj_i + (1.0 - weight) * merits

        # Normalize final merit to [0, 1]
        m_min, m_max = np.min(merits), np.max(merits)
        if m_max - m_min > 1e-8:
            merits = (merits - m_min) / (m_max - m_min)
        else:
            merits = np.zeros_like(merits)

        return merits


class ObservationProcessor:
    """
    Processes observation dictionaries, performs boundary mirroring,
    and scalarizes multi-objective targets.
    """
    def __init__(self, config):
        self.config = config
        self.feature_lowers = config.feature_lowers
        self.feature_uppers = config.feature_uppers
        self.soft_lower = self.feature_lowers + 0.1 * (self.feature_uppers - self.feature_lowers)
        self.soft_upper = self.feature_uppers - 0.1 * (self.feature_uppers - self.feature_lowers)
        
        self.scalarizer = MultiObjectiveScalarizer(
            goals=config.obj_goals,
            tolerances=config.obj_tolerances,
            absolutes=config.obj_absolutes
        )

        self.min_obj = None
        self.max_obj = None

    def mirror_parameters(self, param_vector):
        """
        Mirror continuous parameters near boundaries to handle edge-effects in KDE.
        """
        lower_indices = np.where(param_vector < self.soft_lower)[0]
        upper_indices = np.where(param_vector > self.soft_upper)[0]

        index_dict = {index: 'lower' for index in lower_indices}
        for index in upper_indices:
            index_dict[index] = 'upper'

        params = []
        index_dict_keys = list(index_dict.keys())
        index_dict_values = list(index_dict.values())
        
        # Limit boundary mirroring to max 2^3 = 8 combinations for speed if many variables are near bounds
        max_mirror_vars = min(len(index_dict_keys), 3)
        index_dict_keys = index_dict_keys[:max_mirror_vars]
        index_dict_values = index_dict_values[:max_mirror_vars]

        for index in range(2**len(index_dict_keys)):
            param_copy = param_vector.copy()
            for jndex in range(len(index_dict_keys)):
                if (index // 2**jndex) % 2 == 1:
                    param_index = index_dict_keys[jndex]
                    if index_dict_values[jndex] == 'lower':
                        param_copy[param_index] = self.feature_lowers[param_index] - (param_vector[param_index] - self.feature_lowers[param_index])
                    elif index_dict_values[jndex] == 'upper':
                        param_copy[param_index] = self.feature_uppers[param_index] + (self.feature_uppers[param_index] - param_vector[param_index])
            params.append(param_copy)

        if len(params) == 0:
            params.append(param_vector.copy())
        return params

    def scalarize_objectives(self, objs, transform='sqrt'):
        """Scalarize multi-objective matrix into single objective and apply transform"""
        self.min_obj = np.amin(objs)
        self.max_obj = np.amax(objs)

        scalarized = self.scalarizer.scalarize(objs)

        if transform is None or transform == 'none':
            return scalarized
        elif transform == 'sqrt':
            return np.sqrt(scalarized)
        elif transform == 'cbrt':
            return np.cbrt(scalarized)
        elif transform == 'square':
            return np.square(scalarized)
        else:
            return scalarized

    def process_observations(self, obs_dicts):
        """
        Parses list of observation dicts.
        Returns:
            obs_params: (N, D) array of parameter vectors (including mirrored points)
            obs_objs: (N,) array of scalarized objective values
            obs_feas: (N,) array of feasibility (0.0 for feasible/known, 1.0 for infeasible/unknown)
            mask_kwn: boolean mask for feasible observations
            mask_mirror: boolean mask indicating if point is a mirrored point
        """
        obs_params = []
        raw_objs = []
        obs_feas = []
        mask_kwn = []
        mask_mirror = []

        for obs_dict in obs_dicts:
            param_vector = param_dict_to_vector(obs_dict, self.config.param_names)
            mirrored_params = self.mirror_parameters(param_vector)

            # Extract objective values
            obj_vector = []
            is_infeasible = False
            for obj_name in self.config.obj_names:
                val = obs_dict.get(obj_name, np.nan)
                if val is None or np.isnan(val):
                    is_infeasible = True
                    obj_vector.append(np.nan)
                else:
                    obj_vector.append(float(val))
            obj_vector = np.array(obj_vector)

            # Process mirrored parameters
            for i, param in enumerate(mirrored_params):
                obs_params.append(param)
                raw_objs.append(obj_vector)

                if is_infeasible or np.any(np.isnan(obj_vector)):
                    feas = 1.0  # Infeasible
                    kwn = False
                else:
                    feas = 0.0  # Feasible
                    kwn = True

                mirror = (i != 0)
                obs_feas.append(feas)
                mask_kwn.append(kwn)
                mask_mirror.append(mirror)

        obs_params = np.array(obs_params, dtype=np.float64)
        raw_objs = np.array(raw_objs, dtype=np.float64)
        obs_feas = np.array(obs_feas, dtype=np.float64)
        mask_kwn = np.array(mask_kwn, dtype=bool)
        mask_mirror = np.array(mask_mirror, dtype=bool)

        obs_objs = np.empty(shape=len(raw_objs), dtype=np.float64)
        if len(raw_objs[mask_kwn]) > 0:
            obs_objs_kwn = self.scalarize_objectives(raw_objs[mask_kwn], transform=self.config.obj_transform)
            obs_objs[mask_kwn] = obs_objs_kwn

        if len(raw_objs[~mask_kwn]) > 0:
            obs_objs[~mask_kwn] = np.nan

        return obs_params, obs_objs, obs_feas, mask_kwn, mask_mirror


def param_dict_to_vector(param_dict, param_names):
    return np.array([param_dict[name] for name in param_names], dtype=np.float64)

def param_vector_to_dict(param_vector, param_names):
    return {name: float(param_vector[i]) for i, name in enumerate(param_names)}

def param_dicts_to_vectors(param_dicts, param_names):
    return np.array([[d[name] for name in param_names] for d in param_dicts], dtype=np.float64)

def param_vectors_to_dicts(param_vectors, param_names):
    return [{name: float(vec[i]) for i, name in enumerate(param_names)} for vec in param_vectors]
