import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configuration
DATA_PATH = 'Machine Downtime.csv'
OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

print("Starting Comprehensive EDA Analysis...")
print("=" * 60)

# Load and prepare data
try:
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'], dayfirst=True)
    print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Identify columns
id_cols = ['Machine_ID', 'Assembly_Line_No', 'Date', 'Downtime']
sensor_cols = [c for c in df.columns if c not in id_cols]

print(f"Identified {len(sensor_cols)} sensor columns")
print(f"Machines: {df['Machine_ID'].unique().tolist()}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# 1. GLOBAL OVERVIEW ANALYSIS
print("\nGenerating Global Analysis...")

# Dataset overview
overview_stats = {
    'Total Records': len(df),
    'Date Range (days)': (df['Date'].max() - df['Date'].min()).days,
    'Machines': df['Machine_ID'].nunique(),
    'Failure Rate (%)': (df['Downtime'] == 'Machine_Failure').mean() * 100,
    'Missing Values (%)': df[sensor_cols].isnull().sum().sum() / (len(df) * len(sensor_cols)) * 100
}

# Save overview
with open(os.path.join(OUT_DIR, 'dataset_overview.txt'), 'w') as f:
    f.write("CNC MAINTENANCE DATASET OVERVIEW\n")
    f.write("=" * 40 + "\n\n")
    for key, value in overview_stats.items():
        f.write(f"{key}: {value:.2f}\n")

# Global correlation matrix
print("Computing correlation matrix...")
corr = df[sensor_cols].corr()

# Create correlation heatmap
plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))  # Mask upper triangle
sns.heatmap(
    corr, 
    mask=mask,
    annot=True, 
    fmt='.2f', 
    cmap='RdBu_r',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": .8}
)
plt.title('Sensor Correlation Matrix\n(Lower Triangle Only)', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'global_sensor_correlation_enhanced.png'), dpi=300, bbox_inches='tight')
plt.close()

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.7:  # High correlation threshold
            high_corr_pairs.append({
                'Sensor_1': corr.columns[i],
                'Sensor_2': corr.columns[j],
                'Correlation': corr.iloc[i, j]
            })

if high_corr_pairs:
    pd.DataFrame(high_corr_pairs).to_csv(
        os.path.join(OUT_DIR, 'high_correlation_pairs.csv'), 
        index=False
    )
    print(f"Found {len(high_corr_pairs)} highly correlated sensor pairs (>0.7)")

# 2. FAILURE ANALYSIS
print("\nAnalyzing Failure Patterns...")

failure_dir = os.path.join(OUT_DIR, 'failure_analysis')
os.makedirs(failure_dir, exist_ok=True)

# Failure statistics by machine
failure_stats = df.groupby('Machine_ID').agg({
    'Downtime': [
        lambda x: (x == 'Machine_Failure').sum(),  # Failure count
        lambda x: (x == 'Machine_Failure').mean() * 100,  # Failure rate
        'count'  # Total records
    ]
}).round(2)

failure_stats.columns = ['Failure_Count', 'Failure_Rate_%', 'Total_Records']
failure_stats.to_csv(os.path.join(failure_dir, 'machine_failure_summary.csv'))

# Time-based failure analysis
df['Date_Only'] = df['Date'].dt.date
daily_failures = df.groupby(['Date_Only', 'Machine_ID']).agg({
    'Downtime': lambda x: (x == 'Machine_Failure').sum()
}).reset_index()
daily_failures.columns = ['Date', 'Machine_ID', 'Daily_Failures']

# Monthly failure trends
df['Year_Month'] = df['Date'].dt.to_period('M')
monthly_failures = df.groupby(['Year_Month', 'Machine_ID']).agg({
    'Downtime': lambda x: (x == 'Machine_Failure').sum()
}).reset_index()

# Create failure timeline visualization
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Daily failures
for machine in df['Machine_ID'].unique():
    machine_data = daily_failures[daily_failures['Machine_ID'] == machine]
    axes[0].plot(pd.to_datetime(machine_data['Date']), machine_data['Daily_Failures'], 
                marker='o', label=machine, alpha=0.7)

axes[0].set_title('Daily Failure Occurrences by Machine', fontsize=14)
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Number of Failures')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Monthly failures
monthly_failures['Year_Month_str'] = monthly_failures['Year_Month'].astype(str)
for machine in df['Machine_ID'].unique():
    machine_data = monthly_failures[monthly_failures['Machine_ID'] == machine]
    axes[1].bar(machine_data['Year_Month_str'], machine_data['Downtime'], 
               alpha=0.7, label=machine)

axes[1].set_title('Monthly Failure Count by Machine', fontsize=14)
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Number of Failures')
axes[1].legend()
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(failure_dir, 'failure_timeline_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. SENSOR ANALYSIS BY DOWNTIME STATUS
print("\nPerforming Enhanced Sensor Analysis...")

sensor_analysis_dir = os.path.join(OUT_DIR, 'sensor_analysis')
os.makedirs(sensor_analysis_dir, exist_ok=True)

# Statistical tests for each sensor
statistical_results = []

for col in sensor_cols:
    # Separate data by failure status
    normal_data = df[df['Downtime'] != 'Machine_Failure'][col].dropna()
    failure_data = df[df['Downtime'] == 'Machine_Failure'][col].dropna()
    
    if len(normal_data) > 10 and len(failure_data) > 10:
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(normal_data, failure_data)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(normal_data) - 1) * normal_data.var() + 
                             (len(failure_data) - 1) * failure_data.var()) / 
                            (len(normal_data) + len(failure_data) - 2))
        cohens_d = (failure_data.mean() - normal_data.mean()) / pooled_std
        
        statistical_results.append({
            'Sensor': col,
            'Normal_Mean': normal_data.mean(),
            'Normal_Std': normal_data.std(),
            'Failure_Mean': failure_data.mean(),
            'Failure_Std': failure_data.std(),
            'T_Statistic': t_stat,
            'P_Value': p_value,
            'Cohens_D': cohens_d,
            'Significant': p_value < 0.05,
            'Effect_Size': 'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'
        })

# Save statistical analysis
stats_df = pd.DataFrame(statistical_results)
stats_df.to_csv(os.path.join(sensor_analysis_dir, 'statistical_analysis.csv'), index=False)

# Identify most significant sensors for failure prediction
significant_sensors = stats_df[stats_df['Significant'] == True].sort_values('P_Value')
print(f"Found {len(significant_sensors)} sensors with significant differences between normal/failure states")

# Create comparison plots for top significant sensors
top_sensors = significant_sensors.head(6)['Sensor'].tolist()

if top_sensors:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, sensor in enumerate(top_sensors):
        # Box plot comparison
        sensor_data = df[[sensor, 'Downtime', 'Machine_ID']].dropna()
        sns.boxplot(data=sensor_data, x='Downtime', y=sensor, hue='Machine_ID', ax=axes[i])
        axes[i].set_title(f'{sensor}\n(p-value: {stats_df[stats_df["Sensor"]==sensor]["P_Value"].iloc[0]:.2e})')
        axes[i].tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    plt.savefig(os.path.join(sensor_analysis_dir, 'top_significant_sensors_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

# 4. MACHINE-SPECIFIC ANALYSIS
print("\nGenerating Machine-Specific Analysis...")

for machine_id, group in df.groupby('Machine_ID'):
    machine_dir = os.path.join(OUT_DIR, f'Machine_{machine_id}_detailed')
    os.makedirs(machine_dir, exist_ok=True)
    
    print(f"  Analyzing {machine_id}...")
    
    # Basic statistics
    machine_stats = {
        'Total_Records': len(group),
        'Failure_Count': (group['Downtime'] == 'Machine_Failure').sum(),
        'Failure_Rate_%': (group['Downtime'] == 'Machine_Failure').mean() * 100,
        'Date_Range_Days': (group['Date'].max() - group['Date'].min()).days,
        'Avg_Daily_Records': len(group) / max(1, (group['Date'].max() - group['Date'].min()).days)
    }
    
    # Save machine overview
    with open(os.path.join(machine_dir, 'machine_overview.txt'), 'w') as f:
        f.write(f"MACHINE {machine_id} ANALYSIS\n")
        f.write("=" * 30 + "\n\n")
        for key, value in machine_stats.items():
            f.write(f"{key}: {value:.2f}\n")
    
    # Missing value analysis
    missing_analysis = group[sensor_cols].isnull().sum().to_frame('missing_count')
    missing_analysis['missing_percentage'] = (missing_analysis['missing_count'] / len(group)) * 100
    missing_analysis = missing_analysis.sort_values('missing_percentage', ascending=False)
    missing_analysis.to_csv(os.path.join(machine_dir, 'missing_values_analysis.csv'))
    
    # Descriptive statistics by failure status
    normal_stats = group[group['Downtime'] != 'Machine_Failure'][sensor_cols].describe()
    failure_stats = group[group['Downtime'] == 'Machine_Failure'][sensor_cols].describe()
    
    # Combine statistics
    combined_stats = pd.concat([normal_stats, failure_stats], keys=['Normal_Operation', 'Machine_Failure'])
    combined_stats.to_csv(os.path.join(machine_dir, 'detailed_statistics_by_status.csv'))
    
    # Sensor distribution plots
    n_sensors = len(sensor_cols)
    n_cols = 4
    n_rows = (n_sensors + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, sensor in enumerate(sensor_cols):
        if i < len(axes):
            # Histogram with failure overlay
            normal_data = group[group['Downtime'] != 'Machine_Failure'][sensor].dropna()
            failure_data = group[group['Downtime'] == 'Machine_Failure'][sensor].dropna()
            
            if len(normal_data) > 0:
                axes[i].hist(normal_data, bins=20, alpha=0.7, label='Normal', density=True)
            if len(failure_data) > 0:
                axes[i].hist(failure_data, bins=20, alpha=0.7, label='Failure', density=True)
            
            axes[i].set_title(f'{sensor}', fontsize=10)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(len(sensor_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(machine_dir, 'sensor_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Time series analysis for this machine
    if len(group) > 50:  # Only if enough data points
        # Resample to daily averages for cleaner visualization
        daily_data = group.set_index('Date')[sensor_cols].resample('D').mean()
        
        # Plot time series for top 6 most variable sensors
        sensor_variability = daily_data.std().sort_values(ascending=False)
        top_variable_sensors = sensor_variability.head(6).index.tolist()
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, sensor in enumerate(top_variable_sensors):
            axes[i].plot(daily_data.index, daily_data[sensor], linewidth=1, alpha=0.8)
            
            # Mark failure days
            failure_dates = group[group['Downtime'] == 'Machine_Failure']['Date'].dt.date.unique()
            for failure_date in failure_dates:
                if pd.to_datetime(failure_date) in daily_data.index:
                    axes[i].axvline(x=pd.to_datetime(failure_date), color='red', linestyle='--', alpha=0.7)
            
            axes[i].set_title(f'{sensor} - Time Series')
            axes[i].set_xlabel('Date')
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(machine_dir, 'time_series_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

# 5. GENERATE COMPREHENSIVE REPORT
print("\nGenerating Comprehensive Report...")

report_path = os.path.join(OUT_DIR, 'EDA_COMPREHENSIVE_REPORT.txt')
with open(report_path, 'w') as f:
    f.write("CNC MAINTENANCE SYSTEM - COMPREHENSIVE EDA REPORT\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("EXECUTIVE SUMMARY\n")
    f.write("-" * 20 + "\n")
    f.write(f"Dataset contains {len(df):,} records across {df['Machine_ID'].nunique()} machines\n")
    f.write(f"Overall failure rate: {(df['Downtime'] == 'Machine_Failure').mean()*100:.2f}%\n")
    f.write(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}\n")
    f.write(f"Total sensors monitored: {len(sensor_cols)}\n\n")
    
    # Machine-specific summary
    f.write("MACHINE FAILURE RATES\n")
    f.write("-" * 20 + "\n")
    for machine in df['Machine_ID'].unique():
        machine_data = df[df['Machine_ID'] == machine]
        failure_rate = (machine_data['Downtime'] == 'Machine_Failure').mean() * 100
        f.write(f"{machine}: {failure_rate:.2f}% ({(machine_data['Downtime'] == 'Machine_Failure').sum()} failures)\n")
    
    f.write("\nMOST PREDICTIVE SENSORS\n")
    f.write("-" * 25 + "\n")
    if not significant_sensors.empty:
        for _, row in significant_sensors.head(5).iterrows():
            f.write(f"{row['Sensor']}: p-value={row['P_Value']:.2e}, Effect size={row['Effect_Size']}\n")
    
    f.write("\nHIGH CORRELATION PAIRS\n")
    f.write("-" * 22 + "\n")
    if high_corr_pairs:
        for pair in high_corr_pairs[:5]:
            f.write(f"{pair['Sensor_1']} <-> {pair['Sensor_2']}: {pair['Correlation']:.3f}\n")
    
    f.write("\nRECOMMENDATIONS\n")
    f.write("-" * 15 + "\n")
    f.write("1. Focus predictive models on sensors with significant failure differences\n")
    f.write("2. Consider removing highly correlated sensors to reduce multicollinearity\n")
    f.write("3. Implement more frequent monitoring for machines with higher failure rates\n")
    f.write("4. Investigate temporal patterns in failure occurrences\n")
    f.write("5. Address missing data issues, especially in critical sensors\n")

print("Comprehensive EDA Analysis Complete!")
print("=" * 60)
print(f"Results saved to: {OUT_DIR}/")
print("Key outputs:")
print("  • Global correlation analysis")
print("  • Machine-specific detailed reports")
print("  • Statistical significance testing")
print("  • Failure pattern analysis")
print("  • Time series visualizations")
print("  • Comprehensive summary report")
print(f"\nCheck the '{OUT_DIR}/' directory for all generated files and visualizations.")