
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
from construct_model import bipotent_differentiation

# --- Filter accepted theta sets ---
def filter_accepted_thetas(theta, percentile):
    theta_accepted = theta.nsmallest(int(len(theta) * percentile), 'D')
    return theta_accepted

def marginal_distributions (theta_accepted, condition):

    # ── Model definitions ──────────────────────────────────────────────────────────
    models = [
        ('constant', 'constant', 'CC'),
        ('constant', 'varying',  'CV'),
        ('varying',  'varying',  'VV'),
        ('varying',  'constant', 'VC'),
    ]

    params_per_model = {
        ('constant', 'constant'): ['p_initial', 'alpha', 'mu', 'k_diff_initial',
                                    'rho_n_initial', 'rho_g_initial'],
        ('constant', 'varying'):  ['p_initial', 'alpha', 'mu', 'k_diff_initial',
                                    'rho_n_initial', 'rho_n_gradient', 'rho_g_initial', 'rho_g_gradient'],
        ('varying',  'varying'):  ['p_initial', 'alpha', 'mu', 'k_diff_initial', 'k_diff_gradient',
                                    'rho_n_initial', 'rho_n_gradient', 'rho_g_initial', 'rho_g_gradient'],
        ('varying',  'constant'): ['p_initial', 'alpha', 'mu', 'k_diff_initial', 'k_diff_gradient',
                                    'rho_n_initial', 'rho_g_initial'],
    }

    # ── Column order (9 parameters) ────────────────────────────────────────────────
    all_params = [
        'p_initial', 'alpha', 'mu', 'k_diff_initial', 'k_diff_gradient',
        'rho_n_initial', 'rho_n_gradient', 'rho_g_initial', 'rho_g_gradient'
    ]

    param_ranges = {
        'p_initial'       : (0, 1),
        'alpha'           : (-0.5, 0.5),
        'mu'              : (0, 0.5),
        'k_diff_initial'  : (0, 0.5),
        'k_diff_gradient' : (-0.083, 0.083),
        'rho_n_initial'   : (0, 1),
        'rho_n_gradient'  : (-0.167, 0.167),
        'rho_g_initial'   : (0, 1),
        'rho_g_gradient'  : (-0.167, 0.167),
    }
    # ── Labels ────────────────────────────────────────────────
    acceptance_percentile = "0.2%" #in % format, so if 0.2% then enter 0.2, this is only for labelling files later.

    # ── Layout ─────────────────────────────────────────────────────────────────────
    n_rows  = len(models)      # 4
    n_cols  = len(all_params)  # 9
    cell_sz = 2.2              # inches per cell → each subplot is square

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_sz, n_rows * cell_sz),
        squeeze=False,
    )

    # ── Find the bottom-most active row per column (for x-axis labels) ────────────
    bottom_row_per_col = {}
    for col_idx, param in enumerate(all_params):
        for row_idx, (k_diff_type, prob_type, label) in enumerate(models):
            if param in params_per_model[(k_diff_type, prob_type)]:
                bottom_row_per_col[col_idx] = row_idx  # keeps overwriting → last wins

    # ── Plot ───────────────────────────────────────────────────────────────────────
    for row_idx, (k_diff_type, prob_type, label) in enumerate(models):
        model_name = f"k_diff_type: {k_diff_type}, probability_type: {prob_type}"
        model_params = params_per_model[(k_diff_type, prob_type)]

        df = theta_accepted[
            (theta_accepted['k_diff_type']      == k_diff_type) &
            (theta_accepted['probability_type'] == prob_type)
        ].copy()

        for col_idx, param in enumerate(all_params):
            ax = axes[row_idx, col_idx]
            ax.set_box_aspect(1)

            # ── Blank cell ────────────────────────────────────────────────────────
            if param not in model_params:
                ax.axis('off')
                continue

            # ── Active cell ───────────────────────────────────────────────────────
            lo, hi = param_ranges[param]
            data   = df[param].dropna()

            if len(data) > 1:
                ax.hist(data, bins=30, density=True, alpha=0.6,
                        color='steelblue', edgecolor='black', linewidth=0.4)
                try:
                    kde     = gaussian_kde(data)
                    x_range = np.linspace(lo, hi, 300)
                    ax.plot(x_range, kde(x_range), color='navy', linewidth=1.5)
                except:
                    pass
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=7)

            ax.set_xlim(lo, hi)

            # ── x-axis: only on the bottom active row of each column ──────────────
            if row_idx == bottom_row_per_col.get(col_idx):
                ax.set_xlabel(param, fontsize=13, labelpad=3)
                ax.xaxis.set_tick_params(labelsize=13)
                ax.set_xticks([lo, (lo + hi) / 2, hi])
                ax.xaxis.set_major_formatter(
                    plt.FuncFormatter(lambda v, _: f'{v:.2g}'))
            else:
                ax.set_xticklabels([])
                ax.tick_params(bottom=False)

            # ── y-axis: only on leftmost active column of each row ────────────────
            if col_idx == 0 or not any(
                    all_params[c] in model_params for c in range(col_idx)):
                ax.set_ylabel(label, fontsize=13, fontweight='bold')
            else:
                ax.set_yticklabels([])
                ax.tick_params(left=False)

    # ── Row labels on the left of each row ────────────────────────────────────────
    for row_idx, (_, _, label) in enumerate(models):
        axes[row_idx, 0].set_ylabel(label, fontsize=13, fontweight='bold', labelpad=6)

    fig.suptitle(f"Marginal Posterior Distributions by model, {condition}",
                fontsize=13, fontweight='bold', y=1.01)

    return fig

# --- Plotting empirical data overlaid by simulated data (median parameter set) ---

def sim_over_emp_median(condition, df_neural, df_progenitors_glial, bio_dfs, theta_accepted, canonical_days=[9,15,22]):
    params_per_model = {
                    ('constant', 'constant'): ['p_initial', 'alpha', 'mu', 'k_diff_initial',
                                                'rho_n_initial', 'rho_g_initial'],
                    ('constant', 'varying'):  ['p_initial', 'alpha', 'mu', 'k_diff_initial',
                                                'rho_n_initial', 'rho_n_gradient', 'rho_g_initial', 'rho_g_gradient'],
                    ('varying',  'varying'):  ['p_initial', 'alpha', 'mu', 'k_diff_initial', 'k_diff_gradient',
                                                'rho_n_initial', 'rho_n_gradient', 'rho_g_initial', 'rho_g_gradient'],
                    ('varying',  'constant'): ['p_initial', 'alpha', 'mu', 'k_diff_initial', 'k_diff_gradient',
                                                'rho_n_initial', 'rho_g_initial'],
                }

    medians = {}
    for (k_diff_type, prob_type), param in params_per_model.items():
          model_name = f'{k_diff_type}', f'{prob_type}'
          print(f"\nCondition: {condition} Model: k_diff_type = {k_diff_type}, probability_type = {prob_type} \n parameters are = {param}")
          for p in param:
            median_value = theta_accepted[
              (theta_accepted['k_diff_type']  == k_diff_type) &
              (theta_accepted['probability_type']  == prob_type)
          ][p].median()

            medians.setdefault((condition, model_name), {})
            medians[condition, model_name][p] = median_value

    cell_types = ['progenitors', 'neural', 'glial']
    cell_type_colors = ['#6a3d9aff','#daa520ff', '#1f78b4ff']
    titles = {'progenitors': 'Progenitors', 'neural': 'Neurons', 'glial': 'Glia'}

    # Colour palette
    constant_color = '#2ca02c'   # green   – Constant model simulated
    varying_color  = '#8b008b'   # magenta – Varying model simulated

    fig, axes = plt.subplots(1, 3, figsize=(8, 5), sharey=True)

    for ax, ct, cc in zip(axes, cell_types, cell_type_colors):
        # ── Select empirical data ─────────────────────────────────────────────
        df_tech = (df_neural.copy()
                  if ct == 'neural'
                  else df_progenitors_glial.copy())
        df_bio  = bio_dfs[ct].copy()

        df_tech_condition = df_tech.query(f"condition == '{condition}'")
        df_bio_condition  = df_bio.query(f"condition == '{condition}'")

        # ── Technical replicate dots ──────────────────────────────────────────
        for d in canonical_days:
            vals_tech = df_tech_condition.query("day == @d")[ct].dropna()
            if len(vals_tech):
                xj = d + 0.08 * np.random.randn(len(vals_tech))
                ax.scatter(xj, vals_tech,
                          color=cc, edgecolor='none',
                          s=18, alpha=0.7, zorder=2)

        # ── Boxplots (biological replicate means) ─────────────────────────────
        for d in canonical_days:
            vals_box = df_tech_condition.query("day == @d")[ct].dropna()
            if not len(vals_box):
                continue
            bp = ax.boxplot(vals_box, positions=[d], widths=2,
                            patch_artist=True, showfliers=False, zorder=3)
            for box in bp['boxes']:
                box.set_facecolor(cc)
                box.set_alpha(0.5)
            for median in bp['medians']:
                median.set_color('black')

        # ── Bio replicate dots (top of box) ──────────────────────────────────
        for d in canonical_days:
            vals_box = df_bio_condition.query("day == @d")[ct].dropna()
            if len(vals_box):
                xj_b = d + 0.02 * np.random.randn(len(vals_box))
                ax.scatter(xj_b, vals_box,
                          facecolor='none', edgecolor='black',
                          s=40, linewidth=1.0, zorder=4)

        # ── Simulated readout overlay (Constant & Varying) ────────────────────
        sim_styles = {
            ('constant', 'constant'): dict(color=constant_color, marker='D', zorder=6, linestyle='--'),
            ('constant', 'varying'):  dict(color=constant_color, marker='s', zorder=6, linestyle='-'),
            ('varying',  'varying'):  dict(color=varying_color,  marker='D', zorder=6, linestyle='--'),
            ('varying',  'constant'): dict(color=varying_color,  marker='s', zorder=6, linestyle='-'),
        }

        for model_name, style in sim_styles.items():
            t_vec = np.linspace(9, 22, 100)
            row   = pd.Series(medians[(condition, model_name)])

            # Extract params common to all models
            k_diff_initial = row['k_diff_initial']
            rho_n_initial  = row['rho_n_initial']
            rho_g_initial  = row['rho_g_initial']
            alpha          = row['alpha']
            mu             = row['mu']
            p_initial      = 0.4 if condition == 'dapt' else row['p_initial']

            # Extract gradient params only if present for this model
            k_diff_gradient = row['k_diff_gradient'] if 'k_diff_gradient' in row.index else 0
            rho_n_gradient  = row['rho_n_gradient']  if 'rho_n_gradient'  in row.index else 0
            rho_g_gradient  = row['rho_g_gradient']  if 'rho_g_gradient'  in row.index else 0

            vars_vec = odeint(bipotent_differentiation,
                    [p_initial, 0, 0, 1 - p_initial],
                    t_vec, args=(k_diff_initial, rho_n_initial, rho_g_initial,
                                k_diff_gradient, rho_n_gradient, rho_g_gradient, alpha, mu,
                                model_name[0], model_name[1]))

            # Calculate fractions AFTER solving
            row_sums  = vars_vec.sum(axis=1, keepdims=True)
            fracs_vec = vars_vec / row_sums

            ax.plot(t_vec, fracs_vec[:, cell_types.index(ct)],
                    color=style['color'], linewidth=2,
                    zorder=style['zorder'], linestyle=style['linestyle'])

        # ── Axis styling ──────────────────────────────────────────────────────
        ax.set_title(titles[ct], color=cc)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel('Time (Days)')
        ax.xaxis.set_major_locator(FixedLocator(canonical_days))
        ax.xaxis.set_major_formatter(FixedFormatter(['9', '15', '22']))
        ax.minorticks_off()
        ax.yaxis.set_major_locator(MultipleLocator(0.1))

    axes[0].set_ylabel('Cell Fraction')

    # ── Legend (on last panel) ────────────────────────────────────────────────
    legend_handles = [
        plt.Line2D([0], [0], color=constant_color, linestyle='--',
                  markersize=7, label='k_diff: constant, bias: constant'),
        plt.Line2D([0], [0], color=constant_color, linestyle='-',
                  markersize=7, label='k_diff: constant, bias: varying'),
        plt.Line2D([0], [0], color=varying_color, linestyle='--',
                  markersize=7, label='k_diff: varying, bias: varying'),
        plt.Line2D([0], [0], color=varying_color, linestyle='-',
                  markersize=7, label='k_diff: varying, bias: constant'),
    ]

    axes[-1].legend(handles=legend_handles, loc='upper left', frameon=True, fontsize=10)

    plt.tight_layout()

    return fig



# --- Plotting empirical data overlaid by simulated data (envelope parameter set) ---

def sim_over_emp_envelope (condition, df_neural, df_progenitors_glial, bio_dfs, theta_accepted, canonical_days=[9,15,22], envelope_size=50):

  params_per_model = {
                    ('constant', 'constant'): ['p_initial', 'alpha', 'mu', 'k_diff_initial',
                                                'rho_n_initial', 'rho_g_initial'],
                    ('constant', 'varying'):  ['p_initial', 'alpha', 'mu', 'k_diff_initial',
                                                'rho_n_initial', 'rho_n_gradient', 'rho_g_initial', 'rho_g_gradient'],
                    ('varying',  'varying'):  ['p_initial', 'alpha', 'mu', 'k_diff_initial', 'k_diff_gradient',
                                                'rho_n_initial', 'rho_n_gradient', 'rho_g_initial', 'rho_g_gradient'],
                    ('varying',  'constant'): ['p_initial', 'alpha', 'mu', 'k_diff_initial', 'k_diff_gradient',
                                                'rho_n_initial', 'rho_g_initial'],
                }

  model_combinations = [
    ('constant', 'constant'),
    ('varying', 'varying'),
    ('constant', 'varying'),
    ('varying', 'constant')
  ]

  parameter_sets = {}

  for model_combination in model_combinations:
      parameter_sets[condition, model_combination] = theta_accepted[
          (theta_accepted['condition'] == condition) &
          (theta_accepted['k_diff_type'] == model_combination[0]) &
          (theta_accepted['probability_type'] == model_combination[1])
      ][[
          'condition', 'k_diff_initial', 'rho_n_initial', 'rho_g_initial',
          'k_diff_gradient', 'rho_n_gradient', 'rho_g_gradient', 'alpha', 'mu', 'p_initial'
      ]]
  cell_types = ['progenitors', 'neural', 'glial']
  cell_type_colors = ['#6a3d9aff','#daa520ff', '#1f78b4ff']
  titles     = {'progenitors': 'Progenitors', 'neural': 'Neurons', 'glial': 'Glia'}

  param_cols_env = ['k_diff_initial', 'rho_n_initial', 'rho_g_initial', 'k_diff_gradient', 'rho_n_gradient', 'rho_g_gradient', 'alpha', 'mu', 'p_initial']
  envelope_color = '#2ca02c'

  # Move fig collection outside the loop, return after
  figs = {}

  for model_name in model_combinations:

      fig, axes = plt.subplots(1, 3, figsize=(8, 5), sharey=True)

      for ax, ct, cc in zip(axes, cell_types, cell_type_colors):
          # ── Select empirical data ─────────────────────────────────────────
          df_tech = (df_neural.copy()
                    if ct == 'neural'
                    else df_progenitors_glial.copy())
          df_bio  = bio_dfs[ct].copy()

          df_tech_condition = df_tech.query(f"condition == '{condition}'")
          df_bio_condition  = df_bio.query(f"condition == '{condition}'")

          # ── Technical replicate dots ──────────────────────────────────────
          for d in canonical_days:
              vals_tech = df_tech_condition.query("day == @d")[ct].dropna()
              if len(vals_tech):
                  xj = d + 0.08 * np.random.randn(len(vals_tech))
                  ax.scatter(xj, vals_tech,
                            color=cc, edgecolor='none',
                            s=18, alpha=0.7, zorder=2)

          # ── Boxplots (biological replicate means) ─────────────────────────
          for d in canonical_days:
              vals_box = df_tech_condition.query("day == @d")[ct].dropna()
              if not len(vals_box):
                  continue
              bp = ax.boxplot(vals_box, positions=[d], widths=2,
                              patch_artist=True, showfliers=False, zorder=3)
              for box in bp['boxes']:
                  box.set_facecolor(cc)
                  box.set_alpha(0.5)
              for median in bp['medians']:
                  median.set_color('black')

          # ── Bio replicate dots (top of box) ──────────────────────────────
          for d in canonical_days:
              vals_box = df_bio_condition.query("day == @d")[ct].dropna()
              if len(vals_box):
                  xj_b = d + 0.02 * np.random.randn(len(vals_box))
                  ax.scatter(xj_b, vals_box,
                            facecolor='none', edgecolor='black',
                            s=40, linewidth=1.0, zorder=4)

          # ── Simulated envelope ────────────────────────────────────────────
          for i in range(envelope_size):
              t_vec = np.linspace(9, 22, 100)
              data = pd.DataFrame(parameter_sets[condition, model_name])
              filtered = data[params_per_model[model_name]]  # only columns relevant to this model
              row = filtered.sample(1).iloc[0]

              # Extract params common to all models
              k_diff_initial = row['k_diff_initial']
              rho_n_initial  = row['rho_n_initial']
              rho_g_initial  = row['rho_g_initial']
              alpha          = row['alpha']
              mu             = row['mu']
              p_initial      = 0.4 if condition == 'dapt' else row['p_initial']

              # Extract gradient params only if present for this model
              k_diff_gradient = row['k_diff_gradient'] if 'k_diff_gradient' in row.index else 0
              rho_n_gradient  = row['rho_n_gradient']  if 'rho_n_gradient'  in row.index else 0
              rho_g_gradient  = row['rho_g_gradient']  if 'rho_g_gradient'  in row.index else 0

              vars_vec = odeint(bipotent_differentiation,
                      [p_initial, 0, 0, 1 - p_initial],
                      t_vec, args=(k_diff_initial, rho_n_initial, rho_g_initial,
                                  k_diff_gradient, rho_n_gradient, rho_g_gradient, alpha, mu,
                                  model_name[0], model_name[1]))

              # Calculate fractions AFTER solving
              row_sums = vars_vec.sum(axis=1, keepdims=True)  # sum across columns for each row
              fracs_vec = vars_vec / row_sums                  # divide each element by its row sum

              ax.plot(t_vec, fracs_vec[:, cell_types.index(ct)],
                      color=envelope_color, linewidth=2, alpha=0.1,
                      zorder=1)

          # ── Axis styling ──────────────────────────────────────────────────
          ax.set_title(titles[ct], color=cc)
          ax.set_ylim(-0.05, 1.05)
          ax.grid(True, linestyle='--', alpha=0.6)
          ax.set_xlabel('Time (Days)')
          ax.xaxis.set_major_locator(FixedLocator(canonical_days))
          ax.xaxis.set_major_formatter(FixedFormatter(['9', '15', '22']))
          ax.minorticks_off()
          ax.yaxis.set_major_locator(MultipleLocator(0.1))

      axes[0].set_ylabel('Cell Fraction')

      # ── Legend (on last panel) ────────────────────────────────────────────
      legend_handles = [
          plt.Line2D([0], [0], color=envelope_color, linestyle='-',
                    markersize=7, label=f"k_diff: {model_name[0]}, prob: {model_name[1]}"),
      ]
      axes[-1].legend(handles=legend_handles, loc='upper left', frameon=True, fontsize=10)

      plt.tight_layout()
      figs[model_name] = fig  # store instead of return

  return figs  # return all four figures at once

# --- Plotting neural vs glial bias across all four models

def initial_neural_vs_glial_bias_all_models(condition, theta_accepted):
    model_combinations = [
        ('constant', 'constant'),
        ('constant', 'varying'),
        ('varying',  'varying'),
        ('varying',  'constant')
    ]

    fig, ax = plt.subplots(figsize=(8, 4))

    palette = ['#daa520', '#1f78b4']

    data = [
        theta_accepted[
            (theta_accepted['condition'] == condition) &
            (theta_accepted['k_diff_type'] == model[0]) &
            (theta_accepted['probability_type'] == model[1])
        ][param].values
        for model in model_combinations
        for param in ['rho_n_initial', 'rho_g_initial']
    ]

    parts = ax.violinplot(data, positions=range(len(data)), showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(palette[i % len(palette)])
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
        pc.set_alpha(1)
        for path in pc.get_paths():
            path.vertices[:, 1] = np.clip(path.vertices[:, 1], 0, 1)

    # Make median, min/max lines bolder and black
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(3.5)

    ax.xaxis.set_major_locator(FixedLocator([0.5, 2.5, 4.5, 6.5]))
    ax.xaxis.set_major_formatter(FixedFormatter([
        'Neural Glial\n\nConstant\nConstant',
        'Neural Glial\n\nConstant\nVarying',
        'Neural Glial\n\nVarying\nVarying',
        'Neural Glial\n\nVarying\nConstant'
    ]))

    ax.set_yticks(np.linspace(0, 1, 11))
    ax.minorticks_off()
    ax.set_ylabel('Marginal distribution', fontsize=18)
    ax.set_title(f"Cross-lineage bias comparison across all models, {condition}", fontsize=18)
    ax.tick_params(labelsize=18)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()

    return fig

# --- Plotting neural vs glial bias gradient across all four models

def neural_vs_glial_bias_gradient_all_models(condition, theta_accepted):
    model_combinations = [
        ('constant', 'varying'),
        ('varying',  'varying')
    ]

    fig, ax = plt.subplots(figsize=(8, 4))

    palette = ['#daa520', '#1f78b4']

    data = [
        theta_accepted[
            (theta_accepted['condition'] == condition) &
            (theta_accepted['k_diff_type'] == model[0]) &
            (theta_accepted['probability_type'] == model[1])
        ][param].values
        for model in model_combinations
        for param in ['rho_n_gradient', 'rho_g_gradient']
    ]

    parts = ax.violinplot(data, positions=range(len(data)), showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(palette[i % len(palette)])
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
        pc.set_alpha(1)
        for path in pc.get_paths():
            path.vertices[:, 1] = np.clip(path.vertices[:, 1], -0.166, 0.166)

    # Make median, min/max lines bolder and black
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(3.5)

    ax.xaxis.set_major_locator(FixedLocator([0.5, 2.5]))
    ax.xaxis.set_major_formatter(FixedFormatter([
        'Neural Glial\n\nConstant\nVarying',
        'Neural Glial\n\nVarying\nVarying'
    ]))

    ax.set_yticks(np.linspace(-0.166, 0.166, 11))
    ax.minorticks_off()
    ax.set_ylabel('Marginal distribution', fontsize=18)
    ax.set_title(f"Cross-lineage bias gradient comparison across all models, {condition}", fontsize=18)
    ax.tick_params(labelsize=18)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()

    return fig

# --- Plotting differentiation rate across all four models

def differentiation_rate_all_models(condition, theta_accepted):
    model_combinations = [
        ('constant', 'constant'),
        ('constant', 'varying'),
        ('varying',  'varying'),
        ('varying',  'constant')
    ]

    fig, ax = plt.subplots(figsize=(8, 4))

    palette = ['#6a3d9aff']

    data = [
        theta_accepted[
            (theta_accepted['condition'] == condition) &
            (theta_accepted['k_diff_type'] == model[0]) &
            (theta_accepted['probability_type'] == model[1])
        ]['k_diff_initial'].values
        for model in model_combinations
    ]

    parts = ax.violinplot(data, positions=range(len(data)), showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(palette[i % len(palette)])
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
        pc.set_alpha(1)
        for path in pc.get_paths():
            path.vertices[:, 1] = np.clip(path.vertices[:, 1], 0, 0.5)

    # Make median, min/max lines bolder and black
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(3.5)

    ax.xaxis.set_major_locator(FixedLocator([0, 1, 2, 3]))
    ax.xaxis.set_major_formatter(FixedFormatter([
        'Constant\nConstant',
        'Constant\nVarying',
        'Varying\nVarying',
        'Varying\nConstant'
    ]))

    ax.set_yticks(np.linspace(0, 0.5, 11))
    ax.minorticks_off()
    ax.set_ylabel('Marginal distribution', fontsize=18)
    ax.set_title(f"Differentiation rate across all models, {condition}", fontsize=18)
    ax.tick_params(labelsize=18)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()

    return fig

# --- Plotting differentiation rate gradient across all four models

def differentiation_rate_gradient_all_models(condition, theta_accepted):
    model_combinations = [
        ('varying',  'varying'),
        ('varying',  'constant')
    ]

    fig, ax = plt.subplots(figsize=(8, 4))

    palette = ['#6a3d9aff']

    data = [
        theta_accepted[
            (theta_accepted['condition'] == condition) &
            (theta_accepted['k_diff_type'] == model[0]) &
            (theta_accepted['probability_type'] == model[1])
        ]['k_diff_gradient'].values
        for model in model_combinations
    ]

    parts = ax.violinplot(data, positions=range(len(data)), showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(palette[i % len(palette)])
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
        pc.set_alpha(1)
        for path in pc.get_paths():
            path.vertices[:, 1] = np.clip(path.vertices[:, 1], -0.083, 0.083)

    # Make median, min/max lines bolder and black
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(3.5)

    ax.xaxis.set_major_locator(FixedLocator([0, 1]))
    ax.xaxis.set_major_formatter(FixedFormatter([
        'Varying\nVarying',
        'Varying\nConstant'
    ]))

    ax.set_yticks(np.linspace(-0.083, 0.083, 11))
    ax.minorticks_off()
    ax.set_ylabel('Marginal distribution', fontsize=18)
    ax.set_title(f"Differentiation rate gradient across all models, {condition}", fontsize=18)
    ax.tick_params(labelsize=18)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()

    return fig

# --- Plotting cross-condition posterior comparisons (violin plot) ---

def cross_condition_marginal_comparison_for_all_models(theta_accepted):
  params_models_colors_ranges = {
    'k_diff_initial'  : ([('constant', 'constant'), ('constant', 'varying'), ('varying', 'constant'), ('varying', 'varying')], '#17becf', (0, 0.5)),
    'k_diff_gradient' : ([('varying', 'constant'), ('varying', 'varying')],                                                    '#17becf', (-0.083, 0.083)),
    'rho_n_initial'   : ([('constant', 'constant'), ('constant', 'varying'), ('varying', 'constant'), ('varying', 'varying')], '#daa520', (0, 1)),
    'rho_n_gradient'  : ([('constant', 'varying'), ('varying', 'varying')],                                                    '#daa520', (-0.166, 0.166)),
    'rho_g_initial'   : ([('constant', 'constant'), ('constant', 'varying'), ('varying', 'constant'), ('varying', 'varying')], '#1f78b4', (0, 1)),
    'rho_g_gradient'  : ([('constant', 'varying'), ('varying', 'varying')],                                                    '#1f78b4', (-0.166, 0.166)),
    'alpha'           : ([('constant', 'constant'), ('constant', 'varying'), ('varying', 'constant'), ('varying', 'varying')], '#E8821A', (-0.5, 0.5)),
    'mu'              : ([('constant', 'constant'), ('constant', 'varying'), ('varying', 'constant'), ('varying', 'varying')], '#702963', (0, 0.5)),
  }

  figs = {}
  for param, (models_, color, range_) in params_models_colors_ranges.items():

      fig, ax = plt.subplots(figsize=(14, 6))

      data = [
          theta_accepted[
              (theta_accepted['condition'] == condition) &
              (theta_accepted['k_diff_type'] == model[0]) &
              (theta_accepted['probability_type'] == model[1])
          ][param].values
          for model in models_
          for condition in ['dmso', 'dapt']
      ]

      palette = [mc.to_hex(np.array(mc.to_rgb(color)) * f) for f in [1.0, 0.5]]
      parts = ax.violinplot(data, positions=range(len(data)), showmedians=True)

      for i, pc in enumerate(parts['bodies']):
          pc.set_facecolor(palette[i % len(palette)])
          pc.set_edgecolor('black')
          pc.set_linewidth(1.5)
          pc.set_alpha(1)
          for path in pc.get_paths():
              path.vertices[:, 1] = np.clip(path.vertices[:, 1], range_[0], range_[1])

      # Make median, min/max lines bolder and black
      for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
          vp = parts[partname]
          vp.set_edgecolor('black')
          vp.set_linewidth(3.5)

      if param in ('k_diff_initial', 'alpha', 'mu', 'rho_n_initial', 'rho_g_initial'):
          ax.xaxis.set_major_locator(FixedLocator([0.5, 2.5, 4.5, 6.5]))
          ax.xaxis.set_major_formatter(FixedFormatter([
              f"Untreated NOTCHi\n\n{models_[0][0]}\n{models_[0][1]}",
              f"Untreated NOTCHi\n\n{models_[1][0]}\n{models_[1][1]}",
              f"Untreated NOTCHi\n\n{models_[2][0]}\n{models_[2][1]}",
              f"Untreated NOTCHi\n\n{models_[3][0]}\n{models_[3][1]}"
          ]))
      elif param == 'k_diff_gradient':
          ax.xaxis.set_major_locator(FixedLocator([0.5, 2.5]))
          ax.xaxis.set_major_formatter(FixedFormatter([
              f"Untreated NOTCHi\n\n{models_[0][0]}\n{models_[0][1]}",
              f"Untreated NOTCHi\n\n{models_[1][0]}\n{models_[1][1]}"
          ]))
      else:  # rho_n_gradient, rho_g_gradient
          ax.xaxis.set_major_locator(FixedLocator([0.5, 2.5]))
          ax.xaxis.set_major_formatter(FixedFormatter([
              f"Untreated NOTCHi\n\n{models_[0][0]}\n{models_[0][1]}",
              f"Untreated NOTCHi\n\n{models_[1][0]}\n{models_[1][1]}"
          ]))

      ax.set_ylim(range_[0], range_[1])
      ax.set_yticks(np.linspace(range_[0], range_[1], 11))
      ax.minorticks_off()
      ax.set_ylabel('Marginal distribution', fontsize=18)
      ax.set_title(f"Cross-condition comparison of {param}", fontsize=18)
      ax.tick_params(labelsize=18)
      ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
      ax.set_axisbelow(True)
      plt.tight_layout()
      figs[param] = fig  # store instead of return

  return figs
