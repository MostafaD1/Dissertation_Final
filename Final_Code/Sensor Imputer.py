import os
import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# Regularization Models
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

warnings.filterwarnings('ignore')

# Configuration & Setup
DATA_PATH = 'Machine Downtime.csv'
RESULTS_DIR = 'results/sensor_imputation'
MODELS_DIR = 'results/sensor_models'

# Create directories
for directory in [RESULTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

print("SENSOR IMPUTATION MODEL BUILDER")
print("=" * 60)
print("Purpose: Predict missing sensor values using other sensor correlations")
print("=" * 60)

# Helper Functions
def clean_filename(sensor_name):
    """Clean sensor name for use in filenames (Windows-safe)"""
    return (sensor_name
            .replace('(', '_')
            .replace(')', '')
            .replace('?', '')
            .replace(' ', '_')
            .replace('/', '_')
            .replace('\\', '_')
            .replace(':', '')
            .replace('*', '')
            .replace('"', '')
            .replace('<', '')
            .replace('>', '')
            .replace('|', ''))

# Data Loading and Preprocessing
def load_and_prepare_data():
    """Load and prepare sensor data"""
    print(f"\n[DATA] Loading sensor data from {DATA_PATH}...")
    
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=['Date'], dayfirst=True)
        print(f"[SUCCESS] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print(f"[ERROR] Could not find {DATA_PATH}")
        exit(1)
    
    # Identify sensor columns (exclude identifiers and target)
    exclude_cols = {'Machine_ID', 'Assembly_Line_No', 'Date', 'Downtime'}
    sensor_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"[SENSORS] Found {len(sensor_cols)} sensor columns:")
    for i, sensor in enumerate(sensor_cols, 1):
        print(f"  {i:2d}. {sensor}")
    
    # Focus on sensor data only
    sensor_data = df[sensor_cols].copy()
    
    # Remove rows with missing values for clean training
    initial_rows = len(sensor_data)
    sensor_data = sensor_data.dropna()
    final_rows = len(sensor_data)
    
    print(f"[PREPROCESSING] Removed {initial_rows - final_rows} rows with missing values")
    print(f"[CLEAN DATA] Final dataset: {final_rows} rows")
    
    return sensor_data, sensor_cols

def analyze_sensor_correlations(data, sensor_cols):
    """Analyze correlations between sensors"""
    print(f"\n[ANALYSIS] Analyzing sensor correlations...")
    
    # Compute correlation matrix
    corr_matrix = data.corr()
    
    # Create correlation heatmap
    try:
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                    cmap='coolwarm', center=0, square=True)
        plt.title('Sensor Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'sensor_correlations.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[PLOT] Correlation matrix saved to {RESULTS_DIR}/sensor_correlations.png")
    except Exception as e:
        print(f"[WARNING] Could not create correlation plot: {e}")
        print("[INFO] Correlation analysis will continue without visualization")
    
    # Find highly correlated sensor pairs
    print(f"\n[CORRELATIONS] Highly correlated sensor pairs (|r| > 0.7):")
    high_corr_pairs = []
    for i in range(len(sensor_cols)):
        for j in range(i+1, len(sensor_cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append((sensor_cols[i], sensor_cols[j], corr_val))
                print(f"  {sensor_cols[i]} <-> {sensor_cols[j]}: {corr_val:.3f}")
    
    if not high_corr_pairs:
        print("  No highly correlated pairs found (threshold: |r| > 0.7)")
    
    # Print correlation summary statistics
    print(f"\n[CORRELATION STATS]")
    corr_values = corr_matrix.values
    corr_values = corr_values[np.triu_indices_from(corr_values, k=1)]  # Upper triangle excluding diagonal
    print(f"  Mean correlation: {corr_values.mean():.3f}")
    print(f"  Max correlation: {corr_values.max():.3f}")
    print(f"  Min correlation: {corr_values.min():.3f}")
    print(f"  Std correlation: {corr_values.std():.3f}")
    
    return corr_matrix

# Model Configurations
def get_model_configurations():
    """Define model configurations with regularization parameters"""
    
    models = {
        'Ridge': {
            'model': Ridge(),
            'params': {
                'regressor__alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
            },
            'regularization': 'L2 - Shrinks coefficients smoothly'
        },
        'Lasso': {
            'model': Lasso(),
            'params': {
                'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
            },
            'regularization': 'L1 - Feature selection via sparsity'
        },
        'ElasticNet': {
            'model': ElasticNet(),
            'params': {
                'regressor__alpha': [0.01, 0.1, 1.0, 10.0],
                'regressor__l1_ratio': [0.1, 0.5, 0.7, 0.9]
            },
            'regularization': 'L1 + L2 - Combined benefits'
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=RANDOM_STATE),
            'params': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [5, 10, None],
                'regressor__min_samples_split': [2, 5, 10]
            },
            'regularization': 'Ensemble averaging + tree constraints'
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'regressor__C': [0.1, 1.0, 10.0, 100.0],
                'regressor__epsilon': [0.01, 0.1, 0.2],
                'regressor__kernel': ['rbf', 'linear']
            },
            'regularization': 'Complexity parameter C + epsilon-insensitive'
        },
        'XGBoost': {
            'model': xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
            'params': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [3, 5, 7],
                'regressor__learning_rate': [0.1, 0.2],
                'regressor__reg_alpha': [0, 0.1, 1.0],  # L1 regularization
                'regressor__reg_lambda': [1, 1.5, 2.0]  # L2 regularization
            },
            'regularization': 'L1/L2 regularization + gradient boosting controls'
        }
    }
    
    return models

# Model Training and Evaluation
def train_sensor_imputation_model(data, target_sensor, sensor_cols, models):
    """Train imputation model for a specific sensor"""
    print(f"\n[TRAINING] Building imputation model for: {target_sensor}")
    print("-" * 60)
    
    # Prepare features and target
    feature_cols = [col for col in sensor_cols if col != target_sensor]
    X = data[feature_cols]
    y = data[target_sensor]
    
    print(f"Features: {len(feature_cols)} sensors")
    print(f"Target: {target_sensor}")
    print(f"Samples: {len(X)}")
    
    # 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Results storage
    results = {}
    trained_models = {}
    
    # Train each model
    for model_name, config in models.items():
        print(f"\n  Training {model_name}...")
        
        try:
            # Create pipeline with scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', config['model'])
            ])
            
            # Grid search for hyperparameters
            grid_search = GridSearchCV(
                pipeline, 
                config['params'],
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            
            # Fit model
            grid_search.fit(X_train, y_train)
            
            # Best model
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            # Cross-validation score (overfitting check)
            cv_scores = cross_val_score(
                best_model, X_train, y_train, 
                cv=5, scoring='neg_mean_squared_error'
            )
            cv_rmse_mean = np.sqrt(-cv_scores.mean())
            cv_rmse_std = np.sqrt(cv_scores.std())
            
            # Test predictions
            y_pred = best_model.predict(X_test)
            
            # Metrics
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_mae = mean_absolute_error(y_test, y_pred)
            test_r2 = r2_score(y_test, y_pred)
            
            # Overfitting check
            train_pred = best_model.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            overfitting_ratio = test_rmse / train_rmse
            
            # Store results
            results[model_name] = {
                'best_params': best_params,
                'cv_rmse_mean': cv_rmse_mean,
                'cv_rmse_std': cv_rmse_std,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'overfitting_ratio': overfitting_ratio,
                'regularization': config['regularization']
            }
            
            trained_models[model_name] = best_model
            
            print(f"    {model_name}: Test RMSE = {test_rmse:.4f}, R² = {test_r2:.4f}")
            print(f"       Overfitting Ratio = {overfitting_ratio:.3f} ({'Good' if overfitting_ratio < 1.2 else 'Check' if overfitting_ratio < 1.5 else 'Concerning'})")
            
        except Exception as e:
            print(f"    {model_name} failed: {str(e)}")
            continue
    
    return results, trained_models, X_test, y_test

def evaluate_overfitting(results):
    """Analyze overfitting across all models"""
    print(f"\n[OVERFITTING ANALYSIS]")
    print("-" * 40)
    print("Overfitting Ratio = Test RMSE / Train RMSE")
    print("• < 1.2: Good generalization")
    print("• 1.2-1.5: Acceptable")
    print("• > 1.5: Potential overfitting")
    print()
    
    for model_name, metrics in results.items():
        ratio = metrics['overfitting_ratio']
        status = "Good" if ratio < 1.2 else "Check" if ratio < 1.5 else "Overfitting"
        print(f"{model_name:12}: {ratio:.3f} {status}")

def create_evaluation_plots(results, trained_models, X_test, y_test, target_sensor):
    """Create comprehensive evaluation plots"""
    
    try:
        # Use clean filename for plots
        clean_sensor_name = clean_filename(target_sensor)
        
        # Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # RMSE Comparison
        models = list(results.keys())
        test_rmse = [results[m]['test_rmse'] for m in models]
        
        axes[0,0].bar(models, test_rmse, alpha=0.7, color='skyblue')
        axes[0,0].set_title('Test RMSE Comparison')
        axes[0,0].set_ylabel('RMSE')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # R² Comparison
        r2_scores = [results[m]['test_r2'] for m in models]
        axes[0,1].bar(models, r2_scores, alpha=0.7, color='lightgreen')
        axes[0,1].set_title('R² Score Comparison')
        axes[0,1].set_ylabel('R² Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # Overfitting Analysis
        overfitting_ratios = [results[m]['overfitting_ratio'] for m in models]
        colors = ['green' if r < 1.2 else 'orange' if r < 1.5 else 'red' for r in overfitting_ratios]
        axes[1,0].bar(models, overfitting_ratios, alpha=0.7, color=colors)
        axes[1,0].axhline(y=1.2, color='orange', linestyle='--', label='Acceptable threshold')
        axes[1,0].axhline(y=1.5, color='red', linestyle='--', label='Overfitting threshold')
        axes[1,0].set_title('Overfitting Analysis (Test/Train RMSE)')
        axes[1,0].set_ylabel('Overfitting Ratio')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Best model predictions vs actual
        best_model_name = min(results.keys(), key=lambda x: results[x]['test_rmse'])
        best_model = trained_models[best_model_name]
        y_pred_best = best_model.predict(X_test)
        
        axes[1,1].scatter(y_test, y_pred_best, alpha=0.6)
        axes[1,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1,1].set_xlabel('Actual Values')
        axes[1,1].set_ylabel('Predicted Values')
        axes[1,1].set_title(f'Best Model: {best_model_name}\nPredictions vs Actual')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'{clean_sensor_name}_evaluation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[PLOT] Evaluation plots saved for {target_sensor}")
        
    except Exception as e:
        print(f"[WARNING] Could not create evaluation plots for {target_sensor}: {e}")
        print("[INFO] Model training will continue without visualization")

def save_results(results, target_sensor):
    """Save results to CSV and JSON"""
    
    # Use clean filename for saving
    clean_sensor_name = clean_filename(target_sensor)
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results).T
    df_results.index.name = 'Model'
    df_results = df_results.sort_values('test_rmse')
    
    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, f'{clean_sensor_name}_results.csv')
    df_results.to_csv(csv_path)
    
    # Save detailed JSON
    import json
    json_path = os.path.join(RESULTS_DIR, f'{clean_sensor_name}_detailed_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"[SAVED] Results saved: {csv_path}")
    
    return df_results

# Main Execution
def main():
    """Main execution function"""
    
    # Load and prepare data
    sensor_data, sensor_cols = load_and_prepare_data()
    
    # Analyze correlations
    corr_matrix = analyze_sensor_correlations(sensor_data, sensor_cols)
    
    # Get model configurations
    models = get_model_configurations()
    
    print(f"\n[MODELS] Will train {len(models)} regularized models:")
    for name, config in models.items():
        print(f"  • {name}: {config['regularization']}")
    
    # Train models for key sensors
    all_results = {}
    
    key_sensors = [
        'Hydraulic_Pressure(bar)',
        'Coolant_Temperature', 
        'Spindle_Vibration(?m)',
        'Voltage(volts)'
    ]
    
    print(f"\n[IMPUTATION] Training imputation models for {len(key_sensors)} key sensors...")
    
    for target_sensor in key_sensors:
        if target_sensor in sensor_cols:
            # Train models
            results, trained_models, X_test, y_test = train_sensor_imputation_model(
                sensor_data, target_sensor, sensor_cols, models
            )
            
            # Evaluate overfitting
            evaluate_overfitting(results)
            
            # Create plots
            create_evaluation_plots(results, trained_models, X_test, y_test, target_sensor)
            
            # Save results
            df_results = save_results(results, target_sensor)
            
            # Save best model with clean filename
            best_model_name = df_results.index[0]
            best_model = trained_models[best_model_name]
            clean_sensor_name = clean_filename(target_sensor)
            model_path = os.path.join(MODELS_DIR, f'{clean_sensor_name}_imputer.joblib')
            joblib.dump(best_model, model_path)
            print(f"[SAVED] Best model ({best_model_name}) saved: {model_path}")
            
            all_results[target_sensor] = {
                'best_model': best_model_name,
                'best_rmse': df_results.iloc[0]['test_rmse'],
                'best_r2': df_results.iloc[0]['test_r2'],
                'regularization': df_results.iloc[0]['regularization']
            }
    
    # Final summary
    print(f"\n{'='*60}")
    print("SENSOR IMPUTATION MODEL SUMMARY")
    print(f"{'='*60}")
    
    for sensor, info in all_results.items():
        print(f"\n{sensor}:")
        print(f"  Best Model: {info['best_model']}")
        print(f"  Test RMSE: {info['best_rmse']:.4f}")
        print(f"  R² Score: {info['best_r2']:.4f}")
        print(f"  Regularization: {info['regularization']}")
    
    print(f"\nSUCCESS: Sensor imputation models trained and saved!")
    print(f"Results: {RESULTS_DIR}/")
    print(f"Models: {MODELS_DIR}/")

if __name__ == "__main__":
    main()