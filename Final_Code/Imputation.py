import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

DATA_PATH = 'Machine Downtime.csv'
RESULTS_DIR = os.path.join('results', 'imputation_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- utils ---
def masked_rmse(true_arr, pred_arr, mask_arr):
    t = true_arr[mask_arr]
    p = pred_arr[mask_arr]
    valid = ~np.isnan(t) & ~np.isnan(p)
    if valid.sum() == 0:
        return np.nan
    return np.sqrt(np.mean((t[valid] - p[valid])**2))

print("Building best-method imputed dataset per sensor ...")
df = pd.read_csv(DATA_PATH, parse_dates=['Date'], dayfirst=True).sort_values('Date')
df.set_index('Date', inplace=True)
exclude = {'Machine_ID', 'Assembly_Line_No', 'Downtime'}
sensor_cols = [c for c in df.columns if c not in exclude]

df_imputed = df.copy()
rows = []

# Evaluate three methods per machine+sensor by masking 5% of observed values
for machine in df['Machine_ID'].unique():
    idx = df['Machine_ID'] == machine
    data = df.loc[idx, sensor_cols].copy()

    rng = np.random.default_rng(42)
    mask = pd.DataFrame(False, index=data.index, columns=sensor_cols)
    for col in sensor_cols:
        valid_idx = data[col].dropna().index
        if len(valid_idx) == 0:
            continue
        n_mask = max(1, int(0.05 * len(valid_idx)))
        chosen = rng.choice(valid_idx, size=min(n_mask, len(valid_idx)), replace=False)
        mask.loc[chosen, col] = True

    data_masked = data.copy()
    data_masked[mask] = np.nan

    # candidate imputations
    mean_imp = data_masked.fillna(data.mean())
    interp_imp = data_masked.interpolate(method='time')
    knn_imp = pd.DataFrame(
        KNNImputer(n_neighbors=5).fit_transform(data_masked),
        index=data_masked.index, columns=sensor_cols
    )

    # collect RMSEs
    for col in sensor_cols:
        col_mask = mask[col].values
        true_arr = data[col].to_numpy()
        rmse_mean = masked_rmse(true_arr, mean_imp[col].to_numpy(), col_mask)
        rmse_interp = masked_rmse(true_arr, interp_imp[col].to_numpy(), col_mask)
        rmse_knn = masked_rmse(true_arr, knn_imp[col].to_numpy(), col_mask)
        rows.append({
            'Machine_ID': machine, 'Sensor': col,
            'rmse_mean': rmse_mean, 'rmse_interp': rmse_interp, 'rmse_knn': rmse_knn
        })

    # choose best per sensor for this machine and fill REAL missing accordingly
    best_method = {}
    for col in sensor_cols:
        # last appended row for this machine/sensor has the three rmse values
        triplet = {'mean': np.inf, 'interp': np.inf, 'knn': np.inf}
        for r in rows[::-1]:
            if r['Machine_ID'] == machine and r['Sensor'] == col:
                triplet = {'mean': r['rmse_mean'], 'interp': r['rmse_interp'], 'knn': r['rmse_knn']}
                break
        best = min(triplet, key=lambda k: (np.nan if pd.isna(triplet[k]) else triplet[k]))
        best_method[col] = best

    # Apply the best method column-wise to the real missing values
    fill_df = data.copy()
    for col in sensor_cols:
        if best_method.get(col) == 'interp':
            candidate = data[[col]].copy()
            candidate[col] = candidate[col].interpolate(method='time').fillna(candidate[col].median())
            fill_df[col] = candidate[col]
        elif best_method.get(col) == 'knn':
            temp = data.copy()
            imp = KNNImputer(n_neighbors=5)
            filled = pd.DataFrame(imp.fit_transform(temp), index=temp.index, columns=temp.columns)
            fill_df[col] = filled[col]
        else:
            fill_df[col] = data[col].fillna(data[col].mean())

    df_imputed.loc[idx, sensor_cols] = fill_df.values

# Save outputs
scores = pd.DataFrame(rows)
scores.to_csv(os.path.join(RESULTS_DIR, 'machine_sensor_rmse.csv'), index=False)
df_imputed.reset_index().to_csv(os.path.join(RESULTS_DIR, 'df_imputed.csv'), index=False)
print("Saved: results/imputation_results/df_imputed.csv and machine_sensor_rmse.csv")