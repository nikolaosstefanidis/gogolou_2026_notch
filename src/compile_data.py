
# --- Calculate neural, progenitor, and glial fractions for all technical replicates ---
def calculate_fractions(dfs):
    # Neural fraction = HuCD-positive / total cells
    df_neural = dfs[0][['condition', 'day', 'bio_rep', 'tech_rep']].copy()
    df_neural['neural'] = dfs[0]['total_hucd'] / dfs[0]['total_cells']
    
    # Progenitor and glial fractions
    df_progenitors_glial = dfs[1][['condition', 'day', 'bio_rep', 'tech_rep']].copy()

    # Progenitors = (SOX10-positive – SOX10/S100b double-positive) / total cells
    df_progenitors_glial['progenitors'] = (
        (dfs[1]['total_sox10'] - dfs[1]['total_s100b']) / dfs[1]['total_cells']
    )

    # Glial = SOX10/S100b double-positive / total cells
    df_progenitors_glial['glial'] = dfs[1]['total_s100b'] / dfs[1]['total_cells']

    return df_neural, df_progenitors_glial

# --- Calculate mean value per timepoint per biological replicate ---
def calculate_bio_rep_means(df_neural, df_progenitors_glial):
    bio_dfs = {
        'neural': (
            df_neural.groupby(['condition', 'day', 'bio_rep'])['neural']
            .mean().reset_index()
        ),
        'progenitors': (
            df_progenitors_glial.groupby(['condition', 'day', 'bio_rep'])['progenitors']
            .mean().reset_index()
        ),
        'glial': (
            df_progenitors_glial.groupby(['condition', 'day', 'bio_rep'])['glial']
            .mean().reset_index()
        ),
    }

    return bio_dfs


# --- Extract fraction data for parameter inference ---
def extract_data(df_neural, df_progenitors_glial):
    # --- Input Data ---
    readout_days = [15, 22]
    conditions = ['dmso', 'dapt']

    # --- Nested Dictionary to Store All Results ---
    fractions = {}

    # --- Process Each Condition ---
    for condition in conditions:
        print(f"\nProcessing Condition: {condition.upper()}")

        # Filter DataFrames
        df_N = df_neural[df_neural['condition'] == condition]
        df_PG = df_progenitors_glial[df_progenitors_glial['condition'] == condition]

        # Extract data for all days
        fractions[condition] = {
            'N_obs': {t: df_N[df_N['day'] == t]['neural'].values for t in readout_days},
            'P_obs': {t: df_PG[df_PG['day'] == t]['progenitors'].values for t in readout_days},
            'G_obs': {t: df_PG[df_PG['day'] == t]['glial'].values for t in readout_days}
        }

        # Verification
        print("--- Verification ---")
        for t in readout_days:
            for cell_type in ['N_obs', 'P_obs', 'G_obs']: #, 'X_obs']:
                print(f"Day {t} {cell_type}: {len(fractions[condition][cell_type][t])}")

    return fractions
