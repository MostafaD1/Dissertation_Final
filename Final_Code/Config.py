"""
Configuration file for CNC Failure Prediction System
Centralizes common paths, constants, and configurations
"""

import os

# Data Paths
DATA_PATH = 'Machine Downtime.csv'

# Results Directories
RESULTS_BASE = 'results'
IMPUTATION_RESULTS = os.path.join(RESULTS_BASE, 'imputation_results')
SENSOR_MODELS = os.path.join(RESULTS_BASE, 'sensor_models')
SENSOR_IMPUTATION = os.path.join(RESULTS_BASE, 'sensor_imputation')
FAST_LOSS_ANALYSIS = os.path.join(RESULTS_BASE, 'fast_loss_analysis')
SENSOR_OPTIMIZATION = os.path.join(RESULTS_BASE, 'sensor_optimization_fixed')
MODEL_ANALYSIS = os.path.join(RESULTS_BASE, 'analysis')
TRAINED_MODELS = os.path.join(RESULTS_BASE, 'models')
OUTPUT_DIR = 'outputs'

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Sensor Configuration
SENSOR_DEFAULTS = {
    'Hydraulic_Pressure(bar)': 100.0,
    'Coolant_Pressure(bar)': 5.0,
    'Air_System_Pressure(bar)': 6.5,
    'Coolant_Temperature': 20.0,
    'Hydraulic_Oil_Temperature(?C)': 48.0,
    'Spindle_Bearing_Temperature(?C)': 35.0,
    'Spindle_Vibration(?m)': 1.0,
    'Tool_Vibration(?m)': 25.0,
    'Spindle_Speed(RPM)': 20000.0,
    'Voltage(volts)': 350.0,
    'Torque(Nm)': 25.0,
    'Cutting(kN)': 3.0
}

SENSOR_RANGES = {
    'Hydraulic_Pressure(bar)': (80, 120),
    'Coolant_Pressure(bar)': (3, 8),
    'Air_System_Pressure(bar)': (5, 8),
    'Coolant_Temperature': (15, 35),
    'Hydraulic_Oil_Temperature(?C)': (35, 65),
    'Spindle_Bearing_Temperature(?C)': (25, 50),
    'Spindle_Vibration(?m)': (0.5, 2.0),
    'Tool_Vibration(?m)': (15, 40),
    'Spindle_Speed(RPM)': (15000, 25000),
    'Voltage(volts)': (320, 380),
    'Torque(Nm)': (15, 40),
    'Cutting(kN)': (2, 5)
}

# Data Processing
EXCLUDE_COLS = {'Machine_ID', 'Assembly_Line_No', 'Date', 'Downtime'}

# Machine ID Mapping
MACHINE_ID_MAP = {
    'Makino-L1-Unit1-2013': 'M1',
    'Makino-L2-Unit1-2015': 'M2',
    'Makino-L3-Unit1-2015': 'M3'
}

# Key sensors for analysis
KEY_SENSORS = [
    'Hydraulic_Pressure(bar)',
    'Coolant_Temperature', 
    'Spindle_Vibration(?m)',
    'Voltage(volts)'
]

# All sensor columns
ALL_SENSORS = [
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

# Model Files
MODEL_FILES = {
    'Random Forest': 'randomforest_model.joblib',
    'XGBoost': 'xgboost_model.joblib',
    'CatBoost': 'catboost_model.joblib'
}

# Plotting Configuration
PLOT_STYLE = 'seaborn-v0_8'
PLOT_PALETTE = "husl"
PLOT_DPI = 300

# Utility function to ensure directories exist
def ensure_directories():
    """Create all necessary directories"""
    directories = [
        IMPUTATION_RESULTS,
        SENSOR_MODELS,
        SENSOR_IMPUTATION,
        FAST_LOSS_ANALYSIS,
        SENSOR_OPTIMIZATION,
        MODEL_ANALYSIS,
        TRAINED_MODELS,
        OUTPUT_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Utility function for clean filenames
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