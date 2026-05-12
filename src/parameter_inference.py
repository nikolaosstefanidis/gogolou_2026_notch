
# --- Parameter inference for all four models ---

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import construct_model
from construct_model import bipotent_differentiation
from scipy.stats import uniform


def parameter_inference(fractions, condition, num_iterations=1000000, p_initial_dapt=0.4):

    # --- Define Priors ---
    k_diff_prior      = uniform(loc=0, scale=0.5)   # Uniform[0, 0.5]
    probability_prior = uniform(loc=0, scale=1)      # Uniform[0, 1]
    alpha_prior       = uniform(loc=-0.5, scale=1)
    neuron_death_prior = uniform(loc=0, scale=0.5)
    p_initial_prior   = uniform(loc=0, scale=1)      # Uniform[0, 1] — only used if condition != 'dapt'

    abc_results_dict = {}

    # --- Define model combinations ---
    model_combinations = [
        ('constant', 'constant'),
        ('varying', 'varying'),
        ('constant', 'varying'),
        ('varying', 'constant')
    ]
    print(f"Total model combinations: {len(model_combinations)}")
    for i, (k_diff_type, prob_type) in enumerate(model_combinations, 1):
        print(f"  Model {i}: k_diff={k_diff_type}, probability={prob_type}")

    if condition == 'dapt':
        print(f"  p_initial fixed at {p_initial_dapt} for condition='dapt'")
    else:
        print(f"  p_initial sampled from prior for condition='{condition}'")

    # Set up readout days
    readout_days = [15, 22]

    # Set up time vector
    t_vec = np.unique(np.concatenate([np.linspace(9, 22, 100), readout_days]))

    # Set up readout indices
    readout_indices = {t: np.where(t_vec == t)[0][0] for t in readout_days}

    # Empirical data for ABC
    emp_data = fractions[condition]

    for k_diff_type, prob_type in model_combinations:
        model_name = f"k_diff={k_diff_type}, probability={prob_type}"
        print(f"\n--- Running Model: {model_name} ---")

        results = []

        for i in range(num_iterations):

            # Sample initial parameters (always needed)
            k_diff_initial = k_diff_prior.rvs()
            rho_n_initial, rho_g_initial = construct_model.resample_parameters(probability_prior)
            alpha = alpha_prior.rvs()
            mu    = neuron_death_prior.rvs()

            # p_initial: fixed for dapt, sampled for dmso
            if condition == 'dapt':
                p_initial = p_initial_dapt
            else:
                p_initial = p_initial_prior.rvs()

            # Sample gradients only if needed
            if k_diff_type == 'varying':
                k_diff_gradient_prior = uniform(loc=-k_diff_initial/6, scale=0.5/6)
                k_diff_gradient = k_diff_gradient_prior.rvs()
            else:
                k_diff_gradient = 0.0  # Not used, but needed for function call

            if prob_type == 'varying':
                while True:
                    rho_n_gradient = uniform(loc=-rho_n_initial/6, scale=1/6).rvs()
                    rho_g_gradient = uniform(loc=-rho_g_initial/6, scale=1/6).rvs()
                    rho_n_at_15 = rho_n_initial + rho_n_gradient * 6
                    rho_g_at_15 = rho_g_initial + rho_g_gradient * 6
                    if rho_n_at_15 + rho_g_at_15 <= 1:
                        break
            else:
                rho_n_gradient = rho_g_gradient = 0.0  # Not used, but needed for function call

            recorder = {}  # records final rho_n and rho_g values; populated in-place during integration

            # Run ODE integration
            vars_vec = odeint(
                bipotent_differentiation,
                [p_initial, 0, 0, 1 - p_initial],
                t_vec,
                args=(k_diff_initial, rho_n_initial, rho_g_initial,
                      k_diff_gradient, rho_n_gradient, rho_g_gradient, alpha, mu,
                      k_diff_type, prob_type,
                      recorder)
            )

            # Calculate fractions AFTER solving
            row_sums  = vars_vec.sum(axis=1, keepdims=True)
            fracs_vec = vars_vec / row_sums

            # Extract final values for probabilities and differentiation rate
            rho_n_final  = recorder['rho_n']
            rho_g_final  = recorder['rho_g']
            k_diff_final = recorder['k_diff']

            # Extract readout values
            simulated_readouts = {
                t: fracs_vec[readout_indices[t]].tolist()
                for t in readout_days
            }

            # Calculate distance
            D = construct_model.calculate_distance(
                simulated_readouts,
                emp_data,
                readout_days
            )

            # Store results
            row = [
                k_diff_initial, rho_n_initial, rho_g_initial,
                k_diff_final, rho_n_final, rho_g_final,
                k_diff_gradient, rho_n_gradient, rho_g_gradient, alpha, mu,
                p_initial
            ]

            row += [val for t in readout_days for val in simulated_readouts[t]] + [D]
            results.append(row)

            if (i + 1) % 100000 == 0:
                print(f"  Iteration {i + 1}/{num_iterations}")

        # Convert to DataFrame
        df_temp = pd.DataFrame(
            results,
            columns=[
                'k_diff_initial', 'rho_n_initial', 'rho_g_initial',
                'k_diff_final', 'rho_n_final', 'rho_g_final',
                'k_diff_gradient', 'rho_n_gradient', 'rho_g_gradient', 'alpha', 'mu',
                'p_initial',
                'fP_15', 'fN_15', 'fG_15', 'fU_15',
                'fP_22', 'fN_22', 'fG_22', 'fU_22', 'D'
            ]
        )

        df_temp['condition']        = condition
        df_temp['k_diff_type']      = k_diff_type
        df_temp['probability_type'] = prob_type

        df_temp = df_temp[[
            'k_diff_type', 'probability_type', 'condition',
            'k_diff_initial', 'rho_n_initial', 'rho_g_initial',
            'k_diff_final', 'rho_n_final', 'rho_g_final',
            'k_diff_gradient', 'rho_n_gradient', 'rho_g_gradient', 'alpha', 'mu',
            'p_initial',
            'fP_15', 'fN_15', 'fG_15', 'fU_15',
            'fP_22', 'fN_22', 'fG_22', 'fU_22', 'D'
        ]]

        abc_results_dict[model_name] = df_temp
        print(f"  Complete. Shape: {df_temp.shape}")

    # --- Validate constraints ---
    print("\n=== Constraint Validation ===")
    for model_name, df in abc_results_dict.items():
        print(f"\nModel: {model_name}")
        init_sum  = df['rho_n_initial'] + df['rho_g_initial']
        print(f"  Initial: rho_n + rho_g <= 1 → {(init_sum <= 1.0).mean()*100:.2f}% pass | max sum = {init_sum.max():.4f}")
        final_sum = df['rho_n_final'] + df['rho_g_final']
        print(f"  Final:   rho_n + rho_g <= 1 → {(final_sum <= 1.0).mean()*100:.2f}% pass | max sum = {final_sum.max():.4f}")

    print("\n=== All Model Combinations Complete ===")
    print(f"Created {len(abc_results_dict)} DataFrames:")
    for key, df in abc_results_dict.items():
        print(f"  - {key}: {df.shape}")

    abc_results_combined = pd.concat(abc_results_dict.values(), ignore_index=True)

    print(f"\nCombined Results Shape: {abc_results_combined.shape}")
    print(f"\nFirst few rows:")
    print(abc_results_combined.head(10))

    print(f"\nModel combinations summary:")
    print(abc_results_combined.groupby(['k_diff_type', 'probability_type']).size())

    return abc_results_combined

def calculate_p_initial(theta_accepted):

    # --- Define model combinations ---
    model_combinations = [
        ('constant', 'constant'),
        ('varying', 'varying'),
        ('constant', 'varying'),
        ('varying', 'constant')
    ]

    medians = []
    for model in model_combinations:
        median = theta_accepted[
            (theta_accepted['k_diff_type'] == model[0]) &
            (theta_accepted['probability_type'] == model[1])
        ]['p_initial'].median()
        medians.append(median)

    mean = np.mean(medians)
    return mean

