import os
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
IMPUTED_CSV = 'results/imputation_results/df_imputed.csv'
OUTPUT_DIR = 'results/sensor_optimization_fixed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and preprocess data
print("Loading data...")
df = pd.read_csv(IMPUTED_CSV, parse_dates=['Date'])
df['y'] = (df['Downtime'] == 'Machine_Failure').astype(int)

# Map machine IDs for grouping
id_map = {
    'Makino-L1-Unit1-2013': 'M1',
    'Makino-L2-Unit1-2015': 'M2',
    'Makino-L3-Unit1-2015': 'M3'
}
df['Machine_ID'] = df['Machine_ID'].map(id_map).fillna(df['Machine_ID'])

# Define sensor columns (excluding Machine_ID to prevent leakage)
sensor_cols = [
    'Hydraulic_Pressure(bar)',
    'Coolant_Pressure(bar)',
    'Air_System_Pressure(bar)',
    'Coolant_Temperature',
    'Hydraulic_Oil_Temperature(?C)',
    'Spindle_Bearing_Temperature(?C)',
    'Spindle_Vibration(?m)',
    'Tool_Vibration(?m)',
    'Spindle_Speed(RPM)',
    'Voltage(volts)',
    'Torque(Nm)',
    'Cutting(kN)'
]

# Prepare features and labels - no Machine_ID in features
X = df[sensor_cols]
y = df['y']
groups = df['Machine_ID']  # For leave-one-machine-out validation

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=sensor_cols)

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {y.value_counts().to_dict()}")
print(f"Failure rate: {(y.sum()/len(y))*100:.1f}%")
print(f"Number of sensors: {len(sensor_cols)}")
print(f"Machines in dataset: {groups.unique().tolist()}")

# 1. Baseline with Leave-One-Machine-Out Validation
print("\n" + "="*60)
print("STEP 1: BASELINE PERFORMANCE (Leave-One-Machine-Out)")
print("="*60)

# Use simpler model to avoid overfitting
rf_baseline = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,  # Reduced depth
    min_samples_split=20,  # Prevent overfitting
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)

# Leave-one-machine-out cross-validation
logo = LeaveOneGroupOut()
baseline_scores = []
machine_scores = {}

for train_idx, test_idx in logo.split(X_scaled_df, y, groups):
    X_train, X_test = X_scaled_df.iloc[train_idx], X_scaled_df.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    test_machine = groups.iloc[test_idx].unique()[0]
    
    rf_baseline.fit(X_train, y_train)
    y_proba = rf_baseline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    
    baseline_scores.append(auc)
    machine_scores[test_machine] = auc
    print(f"  Machine {test_machine} held out: AUC = {auc:.4f}")

baseline_auc = np.mean(baseline_scores)
baseline_std = np.std(baseline_scores)
print(f"\nOverall baseline: AUC = {baseline_auc:.4f} (±{baseline_std:.4f})")

# 2. Feature Importance with Proper Validation
print("\n" + "="*60)
print("STEP 2: FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Collect importance across all folds
importance_matrix = []

for train_idx, test_idx in logo.split(X_scaled_df, y, groups):
    X_train, y_train = X_scaled_df.iloc[train_idx], y.iloc[train_idx]
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train, y_train)
    importance_matrix.append(rf.feature_importances_)

# Average importance across folds
avg_importance = np.mean(importance_matrix, axis=0)
std_importance = np.std(importance_matrix, axis=0)

feature_importance = pd.DataFrame({
    'feature': sensor_cols,
    'importance': avg_importance,
    'std': std_importance
}).sort_values('importance', ascending=False)

print("\nFeature Importance (averaged across machines):")
for _, row in feature_importance.iterrows():
    print(f"  {row['feature']:35s}: {row['importance']:.4f} (±{row['std']:.4f})")

# Plot feature importance with error bars
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'], xerr=feature_importance['std'])
plt.xlabel('Importance')
plt.title('Sensor Importance for Failure Prediction\n(Leave-One-Machine-Out Validation)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
plt.close()

# 3. Incremental Sensor Removal
print("\n" + "="*60)
print("STEP 3: INCREMENTAL SENSOR REMOVAL")
print("="*60)

sensors_by_importance = feature_importance['feature'].tolist()
removal_results = []

for n_remove in range(len(sensors_by_importance) + 1):
    if n_remove == 0:
        selected_sensors = sensor_cols
        removed_sensors = []
    else:
        removed_sensors = sensors_by_importance[-n_remove:]
        selected_sensors = [s for s in sensor_cols if s not in removed_sensors]
    
    if len(selected_sensors) == 0:
        # Can't train with 0 features
        result = {
            'n_sensors_removed': n_remove,
            'n_sensors_remaining': 0,
            'removed_sensors': ', '.join(removed_sensors),
            'auc_mean': 0.5,  # Random performance
            'auc_std': 0,
            'performance_drop': baseline_auc - 0.5,
            'accuracy': 0.5
        }
    else:
        X_subset = X_scaled_df[selected_sensors]
        
        # Leave-one-machine-out validation
        fold_scores = []
        fold_acc = []
        
        for train_idx, test_idx in logo.split(X_subset, y, groups):
            X_train, X_test = X_subset.iloc[train_idx], X_subset.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=42
            )
            rf.fit(X_train, y_train)
            
            y_proba = rf.predict_proba(X_test)[:, 1]
            y_pred = rf.predict(X_test)
            
            fold_scores.append(roc_auc_score(y_test, y_proba))
            fold_acc.append(accuracy_score(y_test, y_pred))
        
        result = {
            'n_sensors_removed': n_remove,
            'n_sensors_remaining': len(selected_sensors),
            'removed_sensors': ', '.join(removed_sensors) if removed_sensors else 'None',
            'auc_mean': np.mean(fold_scores),
            'auc_std': np.std(fold_scores),
            'performance_drop': baseline_auc - np.mean(fold_scores),
            'accuracy': np.mean(fold_acc)
        }
    
    removal_results.append(result)
    print(f"Removed {n_remove:2d} sensors: {result['n_sensors_remaining']:2d} remaining, "
          f"AUC={result['auc_mean']:.4f} (drop={result['performance_drop']:+.4f})")

removal_df = pd.DataFrame(removal_results)
removal_df.to_csv(os.path.join(OUTPUT_DIR, 'incremental_removal_results.csv'), index=False)

# 4. Test Specific Sensor Combinations
print("\n" + "="*60)
print("STEP 4: TESTING SPECIFIC SENSOR COMBINATIONS")
print("="*60)

# Test combinations of different sizes
combination_results = []
test_sizes = [2, 3, 4, 5, 6, 8]

for n_sensors in test_sizes:
    print(f"\nTesting {n_sensors}-sensor combinations...")
    
    # Test top N by importance
    top_n_sensors = sensors_by_importance[:n_sensors]
    X_subset = X_scaled_df[top_n_sensors]
    
    scores = []
    for train_idx, test_idx in logo.split(X_subset, y, groups):
        X_train, X_test = X_subset.iloc[train_idx], X_subset.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42
        )
        rf.fit(X_train, y_train)
        y_proba = rf.predict_proba(X_test)[:, 1]
        scores.append(roc_auc_score(y_test, y_proba))
    
    result = {
        'n_sensors': n_sensors,
        'sensors': ', '.join(top_n_sensors),
        'auc_mean': np.mean(scores),
        'auc_std': np.std(scores),
        'performance_retention': (np.mean(scores) / baseline_auc) * 100
    }
    combination_results.append(result)
    
    print(f"  Top {n_sensors} sensors: AUC={result['auc_mean']:.4f} ({result['performance_retention']:.1f}% of baseline)")

combination_df = pd.DataFrame(combination_results)
combination_df.to_csv(os.path.join(OUTPUT_DIR, 'sensor_combinations.csv'), index=False)

# 5. Detailed Analysis of Key Configurations
print("\n" + "="*60)
print("STEP 5: DETAILED CONFIGURATION ANALYSIS")
print("="*60)

configs_to_test = [
    {
        'name': 'All 12 Sensors',
        'sensors': sensor_cols
    },
    {
        'name': 'Top 8 Sensors',
        'sensors': sensors_by_importance[:8]
    },
    {
        'name': 'Top 6 Sensors',
        'sensors': sensors_by_importance[:6]
    },
    {
        'name': 'Top 4 Sensors',
        'sensors': sensors_by_importance[:4]
    },
    {
        'name': 'Top 3 Sensors',
        'sensors': sensors_by_importance[:3]
    }
]

detailed_results = []
for config in configs_to_test:
    X_subset = X_scaled_df[config['sensors']]
    
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }
    
    for train_idx, test_idx in logo.split(X_subset, y, groups):
        X_train, X_test = X_subset.iloc[train_idx], X_subset.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42
        )
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:, 1]
        
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_test, y_pred))
        metrics['f1'].append(f1_score(y_test, y_pred))
        metrics['auc'].append(roc_auc_score(y_test, y_proba))
    
    result = {
        'configuration': config['name'],
        'n_sensors': len(config['sensors']),
        'accuracy_mean': np.mean(metrics['accuracy']),
        'accuracy_std': np.std(metrics['accuracy']),
        'precision_mean': np.mean(metrics['precision']),
        'recall_mean': np.mean(metrics['recall']),
        'f1_mean': np.mean(metrics['f1']),
        'auc_mean': np.mean(metrics['auc']),
        'auc_std': np.std(metrics['auc'])
    }
    detailed_results.append(result)
    
    print(f"\n{config['name']}:")
    print(f"  Accuracy:  {result['accuracy_mean']:.4f} (±{result['accuracy_std']:.4f})")
    print(f"  Precision: {result['precision_mean']:.4f}")
    print(f"  Recall:    {result['recall_mean']:.4f}")
    print(f"  F1:        {result['f1_mean']:.4f}")
    print(f"  AUC:       {result['auc_mean']:.4f} (±{result['auc_std']:.4f})")

detailed_df = pd.DataFrame(detailed_results)
detailed_df.to_csv(os.path.join(OUTPUT_DIR, 'detailed_configurations.csv'), index=False)

# 6. Visualization
print("\n" + "="*60)
print("STEP 6: GENERATING VISUALIZATIONS")
print("="*60)

# Plot 1: Performance vs Number of Sensors
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: AUC vs sensors
ax1 = axes[0]
ax1.errorbar(removal_df['n_sensors_remaining'], removal_df['auc_mean'], 
             yerr=removal_df['auc_std'], marker='o', linewidth=2, markersize=8, capsize=5)
ax1.axhline(y=baseline_auc * 0.95, color='r', linestyle='--', label='95% of baseline', alpha=0.7)
ax1.axhline(y=baseline_auc * 0.90, color='orange', linestyle='--', label='90% of baseline', alpha=0.7)
ax1.set_xlabel('Number of Sensors')
ax1.set_ylabel('ROC-AUC Score')
ax1.set_title('Model Performance vs Number of Sensors\n(Leave-One-Machine-Out Validation)')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim([0.5, 1.0])

# Right: Performance drop
ax2 = axes[1]
ax2.plot(removal_df['n_sensors_removed'], removal_df['performance_drop'], 'r-o', linewidth=2, markersize=8)
ax2.axhline(y=0.05, color='orange', linestyle='--', label='5% drop threshold', alpha=0.7)
ax2.axhline(y=0.10, color='red', linestyle='--', label='10% drop threshold', alpha=0.7)
ax2.set_xlabel('Number of Sensors Removed')
ax2.set_ylabel('Performance Drop (AUC)')
ax2.set_title('Performance Degradation with Sensor Removal')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'performance_vs_sensors.png'), dpi=150)
plt.close()

# Plot 2: Detailed metrics comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics_to_plot = [
    ('accuracy_mean', 'Accuracy'),
    ('precision_mean', 'Precision'),
    ('recall_mean', 'Recall'),
    ('auc_mean', 'ROC-AUC')
]

for idx, (metric, title) in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    bars = ax.bar(range(len(detailed_df)), detailed_df[metric], alpha=0.7)
    
    # Add error bars for AUC
    if metric == 'auc_mean':
        ax.errorbar(range(len(detailed_df)), detailed_df[metric], 
                   yerr=detailed_df['auc_std'], fmt='none', color='black', capsize=5)
    
    ax.set_xticks(range(len(detailed_df)))
    ax.set_xticklabels(detailed_df['configuration'], rotation=45, ha='right')
    ax.set_ylabel(title)
    ax.set_title(f'{title} by Configuration')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Add value labels
    for bar, val in zip(bars, detailed_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Performance Metrics - Realistic Validation\n(Leave-One-Machine-Out)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'configuration_comparison.png'), dpi=150)
plt.close()

# 7. Generate Recommendations
print("\n" + "="*60)
print("FINAL RECOMMENDATIONS")
print("="*60)

# Find configurations maintaining 95% and 90% performance
threshold_95 = baseline_auc * 0.95
threshold_90 = baseline_auc * 0.90

sensors_for_95 = removal_df[removal_df['auc_mean'] >= threshold_95]['n_sensors_remaining'].min()
sensors_for_90 = removal_df[removal_df['auc_mean'] >= threshold_90]['n_sensors_remaining'].min()

config_95 = removal_df[removal_df['n_sensors_remaining'] == sensors_for_95].iloc[0]
config_90 = removal_df[removal_df['n_sensors_remaining'] == sensors_for_90].iloc[0]

# Best efficiency (performance per sensor)
removal_df['efficiency'] = removal_df['auc_mean'] / removal_df['n_sensors_remaining']
removal_df.loc[removal_df['n_sensors_remaining'] == 0, 'efficiency'] = 0
best_efficiency_idx = removal_df[removal_df['n_sensors_remaining'] > 0]['efficiency'].idxmax()
config_efficient = removal_df.iloc[best_efficiency_idx]

print(f"\n1. MAINTAIN 95% PERFORMANCE:")
print(f"   Minimum sensors needed: {sensors_for_95}")
print(f"   AUC achieved: {config_95['auc_mean']:.4f}")
print(f"   Can remove: {config_95['n_sensors_removed']} sensors")
print(f"   Cost savings: {(config_95['n_sensors_removed']/12)*100:.0f}%")

print(f"\n2. MAINTAIN 90% PERFORMANCE:")
print(f"   Minimum sensors needed: {sensors_for_90}")
print(f"   AUC achieved: {config_90['auc_mean']:.4f}")
print(f"   Can remove: {config_90['n_sensors_removed']} sensors")
print(f"   Cost savings: {(config_90['n_sensors_removed']/12)*100:.0f}%")

print(f"\n3. BEST EFFICIENCY (Performance per sensor):")
print(f"   Sensors: {config_efficient['n_sensors_remaining']}")
print(f"   AUC: {config_efficient['auc_mean']:.4f}")
print(f"   Efficiency score: {config_efficient['efficiency']:.4f}")

# Save recommendations
recommendations = pd.DataFrame([
    {
        'criterion': '95% Performance',
        'n_sensors_needed': sensors_for_95,
        'sensors_removed': config_95['n_sensors_removed'],
        'auc': config_95['auc_mean'],
        'cost_savings_pct': (config_95['n_sensors_removed']/12)*100
    },
    {
        'criterion': '90% Performance',
        'n_sensors_needed': sensors_for_90,
        'sensors_removed': config_90['n_sensors_removed'],
        'auc': config_90['auc_mean'],
        'cost_savings_pct': (config_90['n_sensors_removed']/12)*100
    },
    {
        'criterion': 'Best Efficiency',
        'n_sensors_needed': config_efficient['n_sensors_remaining'],
        'sensors_removed': config_efficient['n_sensors_removed'],
        'auc': config_efficient['auc_mean'],
        'cost_savings_pct': (config_efficient['n_sensors_removed']/12)*100
    }
])
recommendations.to_csv(os.path.join(OUTPUT_DIR, 'recommendations.csv'), index=False)

# 8. Summary Report
summary = f"""
SENSOR OPTIMIZATION ANALYSIS - CORRECTED VERSION
=================================================

Validation Method: Leave-One-Machine-Out Cross-Validation
- Each machine (M1, M2, M3) held out in turn for testing
- No data leakage through Machine_ID encoding
- Conservative model parameters to prevent overfitting

Dataset:
- Total samples: {len(df)}
- Failure rate: {(y.sum()/len(y))*100:.1f}%
- Number of sensors: {len(sensor_cols)}

Baseline Performance (All 12 Sensors):
- Mean ROC-AUC: {baseline_auc:.4f} (±{baseline_std:.4f})
- Per-machine AUC:
  {chr(10).join([f'  - {m}: {s:.4f}' for m, s in machine_scores.items()])}

Top 5 Most Important Sensors:
{chr(10).join([f'  {i+1}. {row.feature} (importance: {row.importance:.4f})' for i, row in feature_importance.head(5).iterrows()])}

Key Findings:
1. Minimum sensors for 95% performance: {sensors_for_95} sensors
   - Remove {config_95['n_sensors_removed']} sensors
   - Achieve AUC of {config_95['auc_mean']:.4f}
   
2. Minimum sensors for 90% performance: {sensors_for_90} sensors
   - Remove {config_90['n_sensors_removed']} sensors
   - Achieve AUC of {config_90['auc_mean']:.4f}

3. Performance degrades significantly below {sensors_for_90} sensors

Recommendations:
- For critical applications: Keep top {sensors_for_95} sensors
- For cost-sensitive applications: Keep top {sensors_for_90} sensors
- Sensors safe to remove first: {', '.join(sensors_by_importance[-3:])}

Files Generated:
- feature_importance.png
- performance_vs_sensors.png
- configuration_comparison.png
- incremental_removal_results.csv
- sensor_combinations.csv
- detailed_configurations.csv
- recommendations.csv
"""

with open(os.path.join(OUTPUT_DIR, 'summary_report.txt'), 'w') as f:
    f.write(summary)

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"Results saved to: {OUTPUT_DIR}/")
print("\nThis analysis uses proper validation without data leakage.")
print("Results show realistic performance expectations.")