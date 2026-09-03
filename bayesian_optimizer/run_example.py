import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import time
from bayesian_optimizer import BayesianOptimizer

def main():
    print("=" * 70)
    print(" Gryffin-Inspired Bayesian Optimization Model Demonstration")
    print(" 8 Continuous Features | 5 Target Objectives | Known Constraints")
    print("=" * 70)

    # 1. Define configuration for 8 experimental features
    parameters = [
        {"name": "temperature", "type": "continuous", "low": 20.0, "high": 100.0},
        {"name": "ph", "type": "continuous", "low": 2.0, "high": 12.0},
        {"name": "pressure", "type": "continuous", "low": 1.0, "high": 25.0},
        {"name": "conc1", "type": "continuous", "low": 0.1, "high": 5.0},
        {"name": "conc2", "type": "continuous", "low": 0.05, "high": 2.0},
        {"name": "flow_rate", "type": "continuous", "low": 0.5, "high": 10.0},
        {"name": "time", "type": "continuous", "low": 10.0, "high": 120.0},
        {"name": "agit_speed", "type": "continuous", "low": 100.0, "high": 1000.0},
    ]

    # 2. Define configuration for 5 target objectives
    objectives = [
        {"name": "yield", "goal": "max", "tolerance": 0.05},
        {"name": "purity", "goal": "max", "tolerance": 0.02},
        {"name": "cost", "goal": "min", "tolerance": 0.10},
        {"name": "byproduct", "goal": "min", "tolerance": 0.05},
        {"name": "energy", "goal": "min", "tolerance": 0.10},
    ]

    config = {
        "general": {
            "sampling_strategies": 2,  # 1 exploitation (λ > 0), 1 exploration (λ < 0)
            "batches": 2,              # 2 recommended samples per iteration
            "feas_approach": "fwa",
            "obj_transform": "sqrt",
            "num_random_samples": 20,
            "random_seed": 42
        },
        "model": {
            "num_epochs": 50,          # Fast demonstration training
            "learning_rate": 0.05,
            "num_draws": 30,
            "hidden_shape": 8,
            "num_layers": 3
        },
        "parameters": parameters,
        "objectives": objectives
    }

    # 3. Define known experimental constraint function
    def custom_constraints(p):
        # p can be a dict (from user/optimizer)
        temp = p['temperature']
        press = p['pressure']
        ph = p['ph']

        # Constraint 1: Safe operation boundary (temperature * pressure <= 1800)
        if temp * press > 1800.0:
            return False
        # Constraint 2: Chemical stability (pH >= 3.5)
        if ph < 3.5:
            return False
        return True

    # 4. Instantiate Bayesian Optimizer
    optimizer = BayesianOptimizer(config_dict=config, known_constraints=custom_constraints)

    # Synthetic experiment simulator function
    def run_experiment(p):
        # Simple synthetic response surface
        t, ph_val, pr, c1, c2, fl, tm, ag = (
            p['temperature'], p['ph'], p['pressure'], p['conc1'],
            p['conc2'], p['flow_rate'], p['time'], p['agit_speed']
        )
        
        # Infeasible process condition simulator (e.g., thermal decomposition if temp > 95 and ph > 10)
        if t > 95.0 and ph_val > 10.0:
            return {"yield": np.nan, "purity": np.nan, "cost": np.nan, "byproduct": np.nan, "energy": np.nan}

        y_yield = 50.0 + 0.5 * t - 0.003 * (t - 60)**2 + 2.0 * ph_val + 5.0 * c1 - 0.1 * pr
        y_purity = 80.0 + 1.5 * ph_val - 0.05 * (tm - 60)**2
        y_cost = 10.0 + 0.2 * pr + 5.0 * c1 + 10.0 * c2
        y_byproduct = 15.0 - 0.1 * t + 0.05 * (ph_val - 7)**2 + 2.0 * c2
        y_energy = 5.0 + 0.05 * t + 0.01 * ag + 0.1 * tm

        return {
            "yield": float(y_yield),
            "purity": float(y_purity),
            "cost": float(y_cost),
            "byproduct": float(y_byproduct),
            "energy": float(y_energy)
        }

    # 5. Optimization Loop
    observations = []
    num_iterations = 4

    print("\nStarting Optimization Loop...")
    for iter_idx in range(num_iterations):
        t0 = time.time()
        print(f"\n--- Iteration {iter_idx + 1} ---")
        
        # Get recommendations
        recommendations = optimizer.recommend(observations=observations)
        t_rec = time.time() - t0
        print(f"Proposed {len(recommendations)} candidate experimental sets (computed in {t_rec:.2f}s):")

        for rec_idx, rec in enumerate(recommendations):
            # Check constraint
            is_valid = custom_constraints(rec)
            print(f"  Sample #{rec_idx + 1}: Temp={rec['temperature']:.1f}°C, pH={rec['ph']:.2f}, Press={rec['pressure']:.1f}bar | Constraint Valid: {is_valid}")
            
            # Evaluate synthetic experiment
            results = run_experiment(rec)
            obs = {**rec, **results}
            observations.append(obs)
            
            yield_val = results['yield']
            yield_str = f"{yield_val:.2f}" if np.isfinite(yield_val) else "NaN (Infeasible)"
            print(f"    --> Measured Yield: {yield_str}")

    print("\n" + "=" * 70)
    print(f" Optimization Completed Successfully! Total Observations: {len(observations)}")
    print("=" * 70)

if __name__ == "__main__":
    main()
