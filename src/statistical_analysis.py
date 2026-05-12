

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from scipy.integrate import odeint
from scipy.stats import gaussian_kde, uniform
from scipy.spatial.distance import jensenshannon
from matplotlib.ticker import FixedLocator, FixedFormatter, MultipleLocator
from tqdm import tqdm
import statistics

def descriptive_statistics (theta_accepted, condition):
  medians_iqrs = []

  params_models = {
      'p_initial'       : [('constant', 'constant'), ('constant', 'varying'), ('varying', 'constant'), ('varying', 'varying')],
      'alpha'           : [('constant', 'constant'), ('constant', 'varying'), ('varying', 'constant'), ('varying', 'varying')],
      'mu'              : [('constant', 'constant'), ('constant', 'varying'), ('varying', 'constant'), ('varying', 'varying')],
      'k_diff_initial'  : [('constant', 'constant'), ('constant', 'varying'), ('varying', 'constant'), ('varying', 'varying')],
      'k_diff_gradient' : [('varying', 'constant'), ('varying', 'varying')],
      'rho_n_initial'   : [('constant', 'constant'), ('constant', 'varying'), ('varying', 'constant'), ('varying', 'varying')],
      'rho_n_gradient'  : [('constant', 'varying'), ('varying', 'varying')],
      'rho_g_initial'   : [('constant', 'constant'), ('constant', 'varying'), ('varying', 'constant'), ('varying', 'varying')],
      'rho_g_gradient'  : [('constant', 'varying'), ('varying', 'varying')],
  }

  for param, models in params_models.items():
      for model in models:
          data = theta_accepted[
              (theta_accepted['condition'] == condition) &
              (theta_accepted['k_diff_type'] == model[0]) &
              (theta_accepted['probability_type'] == model[1])
          ][param]

          q1 = np.percentile(data, 25)
          q3 = np.percentile(data, 75)
          iqr = q3 - q1
          median = data.median()
          mean = data.mean()
          sd = statistics.stdev(data)
          n = len(data)

          medians_iqrs.append({
              'NOTCHi' : 'TRUE' if condition == 'dapt' else 'FALSE',
              'Differentiation rate' : model[0],
              'Lineage bias': model[1],
              'parameter' : param,
              'Mean'      : mean,
              'SD'        : sd,
              'Median'    : median,
              'IQR'       : iqr,
              'N'         : n,
              })

  results_df = pd.DataFrame(medians_iqrs)
  return results_df

def _compute_js(samples_1, samples_2, param_ranges, n_points=500):
    """Core JS computation — no permutation loop."""
    low, high = param_ranges
    s1 = (samples_1 - low) / (high - low)
    s2 = (samples_2 - low) / (high - low)

    grid = np.linspace(0, 1, n_points)
    p = gaussian_kde(s1)(grid)
    q = gaussian_kde(s2)(grid)

    p = p + p.max() * 1e-6
    q = q + q.max() * 1e-6
    p /= p.sum()
    q /= q.sum()

    return jensenshannon(p, q), p, q  # returns JS distance, and densities


def compare_posteriors(samples_1, samples_2, param, param_ranges, model=None, n_points=500, n_permutations=10000):
    low, high = param_ranges

    # Observed metrics
    js, p, q = _compute_js(samples_1, samples_2, param_ranges, n_points=500)
    js_div = js ** 2

    # Observed JS on coarse grid — for fair comparison with null
    js_coarse, _, _ = _compute_js(samples_1, samples_2, param_ranges, n_points=50)

    # Permutation test — calls _compute_js, NOT compare_posteriors
    combined = np.concatenate([samples_1, samples_2])
    n1 = len(samples_1)

    null_js = []
    for _ in tqdm(range(n_permutations), desc=f"{param} {model}", leave=False):
        np.random.shuffle(combined)
        js_perm, _, _ = _compute_js(combined[:n1], combined[n1:], param_ranges, n_points=50)
        null_js.append(js_perm)

    null_js = np.array(null_js)
    N_extreme = (null_js >= js_coarse).sum()
    p_value = (N_extreme + 1) / (n_permutations + 1)

    return {
        'JS_distance':  js,
        'JS_divergence': js_div,
        'null_mean':    null_js.mean(),
        'null_std':     null_js.std(),
        'z_score':      (js_coarse - null_js.mean()) / null_js.std(),
        'p_value':      p_value,
        }

def cross_condition_js(theta_accepted):
  params_models_ranges = {
    'k_diff_initial'  : ([('constant', 'constant'), ('constant', 'varying'), ('varying', 'constant'), ('varying', 'varying')], (0, 0.5)),
    'k_diff_gradient' : ([('varying', 'constant'), ('varying', 'varying')],                                                    (-0.083, 0.083)),
    'rho_n_initial'   : ([('constant', 'constant'), ('constant', 'varying'), ('varying', 'constant'), ('varying', 'varying')], (0, 1)),
    'rho_n_gradient'  : ([('constant', 'varying'), ('varying', 'varying')],                                                    (-0.166, 0.166)),
    'rho_g_initial'   : ([('constant', 'constant'), ('constant', 'varying'), ('varying', 'constant'), ('varying', 'varying')], (0, 1)),
    'rho_g_gradient'  : ([('constant', 'varying'), ('varying', 'varying')],                                                    (-0.166, 0.166)),
    'alpha'           : ([('constant', 'constant'), ('constant', 'varying'), ('varying', 'constant'), ('varying', 'varying')], (-0.5, 0.5)),
    'mu'              : ([('constant', 'constant'), ('constant', 'varying'), ('varying', 'constant'), ('varying', 'varying')], (0, 0.5)),
  }

  results = {}

  for param, (models, ranges) in params_models_ranges.items():
      print(f"\n dmso vs dapt for {param}")

      for model in models:
          group_1 = theta_accepted[
              (theta_accepted['condition'] == 'dmso') &
              (theta_accepted['k_diff_type'] == model[0]) &
              (theta_accepted['probability_type'] == model[1])
          ][param]

          group_2 = theta_accepted[
              (theta_accepted['condition'] == 'dapt') &
              (theta_accepted['k_diff_type'] == model[0]) &
              (theta_accepted['probability_type'] == model[1])
          ][param]

          if len(group_1) < 10 or len(group_2) < 10:
              print(f"  WARNING: Too few samples for {model} — dmso: {len(group_1)}, dapt: {len(group_2)}")
              continue

          print(f"  Model {model}: n_dmso={len(group_1)}, n_dapt={len(group_2)}, "
                f"bw_dmso={gaussian_kde(group_1).factor:.4f}, bw_dapt={gaussian_kde(group_2).factor:.4f}")

          results[(param, model)] = compare_posteriors(
              group_1.values, group_2.values, param, ranges, model=model
          )

  results_df = pd.DataFrame(results).T
  results_df.index.names = ['param', 'model']
  return results_df

def cross_lineage_js(theta_accepted, condition):
  combinations_models_ranges = {
    ('rho_n_initial', 'rho_g_initial') : ([('constant', 'constant'), ('constant', 'varying'), ('varying', 'constant'), ('varying', 'varying')], (0, 1)),
    ('rho_n_gradient', 'rho_g_gradient') : ([('constant', 'varying'), ('varying', 'varying')],                                                  (-0.166, 0.166))
  }

  results = {}

  for combination, (models, ranges) in combinations_models_ranges.items():
      print(f"\n{combination[0]} vs {combination[1]} in {condition}")

      for model in models:
          group_1 = theta_accepted[
              (theta_accepted['condition'] == condition) &
              (theta_accepted['k_diff_type'] == model[0]) &
              (theta_accepted['probability_type'] == model[1])
          ][combination[0]]

          group_2 = theta_accepted[
              (theta_accepted['condition'] == condition) &
              (theta_accepted['k_diff_type'] == model[0]) &
              (theta_accepted['probability_type'] == model[1])
          ][combination[1]]

          if len(group_1) < 10 or len(group_2) < 10:
              print(f"  WARNING: Too few samples for {model} — group_1: {len(group_1)}, group_2: {len(group_2)}")
              continue

          print(f"  Model {model}: n_{combination[0]}={len(group_1)}, n_{combination[1]}={len(group_2)}, "
                f"bw_{combination[0]}={gaussian_kde(group_1).factor:.4f}, bw_{combination[1]}={gaussian_kde(group_2).factor:.4f}")

          results[(condition, combination[0], combination[1], model)] = compare_posteriors(
              group_1.values, group_2.values, f"{combination[0]}_vs_{combination[1]}", ranges, model=model
          )

  results_df = pd.DataFrame(results).T
  results_df.index.names = ['condition', 'parameter_1', 'parameter_2', 'model']
  return results_df

