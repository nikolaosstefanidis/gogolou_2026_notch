
# --- Core evolution functions ---

import numpy as np

# Integration parameters, for parameter evolution
t_start, t_final = 9.0, 22.0
decay_duration = t_final - t_start  # 13.0

def calculate_k_diff_t(t, k_diff_initial, k_diff_gradient):
    """
    Changes k_diff linearly with given gradient.
    Lower bound at 0, no upper limit.
    """
    t_relative = t - t_start
    value = k_diff_initial + k_diff_gradient * t_relative
    return max(0.0, value)


def calculate_rho_n_rho_g_t(t, rho_n_initial, rho_g_initial, rho_n_gradient, rho_g_gradient):
    t_relative = t - t_start

    # Calculate unconstrained values
    rho_n_unconstrained = rho_n_initial + rho_n_gradient * t_relative
    rho_g_unconstrained = rho_g_initial + rho_g_gradient * t_relative

    # Check sum constraint for all gradient combinations
    if rho_n_unconstrained + rho_g_unconstrained > 1.0:
        # Find the time when sum would hit 1
        gradient_sum = rho_n_gradient + rho_g_gradient
        if gradient_sum != 0:  # avoid division by zero if gradients cancel out
            t_hit = (1.0 - rho_n_initial - rho_g_initial) / gradient_sum
            rho_n_t = rho_n_initial + rho_n_gradient * t_hit
            rho_g_t = rho_g_initial + rho_g_gradient * t_hit
        else:
            # Gradients cancel — sum is constant, no freeze needed
            rho_n_t = rho_n_unconstrained
            rho_g_t = rho_g_unconstrained
    else:
        rho_n_t = rho_n_unconstrained
        rho_g_t = rho_g_unconstrained

    # Apply individual bounds [0, 1]
    rho_n_t = np.clip(rho_n_t, 0.0, 1.0)
    rho_g_t = np.clip(rho_g_t, 0.0, 1.0)

    return rho_n_t, rho_g_t

# --- Probability resampling if rho_n(9) + rho_g(9) > 1 ---
def resample_parameters(probability_prior):
    """Resample rho_n and rho_g until their sum is <= 1."""
    while True:
        rho_n_initial, rho_g_initial = probability_prior.rvs(2)
        if rho_n_initial + rho_g_initial <= 1.0:
            return rho_n_initial, rho_g_initial

# --- bipotent_differentiation function ---
def bipotent_differentiation(y, t,
                            k_diff_initial, rho_n_initial, rho_g_initial,
                            k_diff_gradient, rho_n_gradient, rho_g_gradient, alpha, mu,
                            k_diff_change_type, probability_change_type,
                            param_recorder=None):
    """
    Unified differentiation function with selectable change types.

    Parameters:
    -----------
    y : array
        [P, N, G, U] population values
    t : float
        Current time
    k_diff_initial, rho_n_initial, rho_g_initial : float
        Initial parameter values
    k_diff_gradient, rho_n_gradient, rho_g_gradient : float
        Gradient parameters (ignored if change_type is 'constant')
    alpha : float
        Net growth rate [negative(death), positive(growth)]
    mu : float
        Neuron explicit death rate
    """
    P, N, G, U = y

    # Handle k_diff
    if k_diff_change_type == 'varying':
        k_diff_t = calculate_k_diff_t(t, k_diff_initial, k_diff_gradient)
    else:  # 'constant'
        k_diff_t = k_diff_initial

    # Handle rho_n & rho_g
    if probability_change_type == 'varying':
        rho_n_t, rho_g_t = calculate_rho_n_rho_g_t(t, rho_n_initial, rho_g_initial, rho_n_gradient, rho_g_gradient)
    else:  # 'constant'
        rho_n_t = rho_n_initial
        rho_g_t = rho_g_initial

    # Always overwrite — after integration, dict holds the last t evaluated
    if param_recorder is not None:
        param_recorder['t']      = t
        param_recorder['rho_n']  = rho_n_t
        param_recorder['rho_g']  = rho_g_t
        param_recorder['k_diff'] = k_diff_t

    return [-k_diff_t * P + alpha * P,
            k_diff_t * rho_n_t * P - mu * N,
            k_diff_t * rho_g_t * P + alpha * G,
            k_diff_t * (1 - rho_n_t - rho_g_t) * P + alpha * U]

def calculate_distance(simulated_readouts, empirical_data, readout_days):
  D = 0.0
  for t in readout_days:
    fP_sim, fN_sim, fG_sim, _ = simulated_readouts[t]

    # Concatenate observations and simulated values
    obs_N = empirical_data['N_obs'].get(t, np.array([]))
    obs_G = empirical_data['G_obs'].get(t, np.array([]))
    obs_P = empirical_data['P_obs'].get(t, np.array([]))

    sim_N = np.full(len(empirical_data['N_obs'].get(t, [])), fN_sim)
    sim_G = np.full(len(empirical_data['G_obs'].get(t, [])), fG_sim)
    sim_P = np.full(len(empirical_data['P_obs'].get(t, [])), fP_sim)

    D += ((np.sum((obs_N - sim_N)**2))/len(obs_N) +
          (np.sum((obs_G - sim_G)**2))/len(obs_G) +
          (np.sum((obs_P - sim_P)**2))/len(obs_P))

  return D
