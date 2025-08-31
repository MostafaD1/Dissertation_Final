import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title='CNC Failure Prediction System',
    page_icon='🔧',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .failure-warning {
        background-color: #d32f2f;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #b71c1c;
    }
    .normal-status {
        background-color: #2e7d32;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1b5e20;
    }
    .confidence-high {
        background-color: #1976d2;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0d47a1;
    }
    .confidence-medium {
        background-color: #f57c00;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #e65100;
    }
    .confidence-low {
        background-color: #d32f2f;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #b71c1c;
    }
</style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<h1 class="main-header">CNC Failure Prediction System</h1>', unsafe_allow_html=True)

# Load Models Function
@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    model_dir = 'results/models'
    
    model_files = {
        'Random Forest': 'randomforest_model.joblib',
        'XGBoost': 'xgboost_model.joblib',
        'CatBoost': 'catboost_model.joblib'
    }
    
    for name, filename in model_files.items():
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            try:
                models[name] = joblib.load(filepath)
                st.sidebar.success(f"{name} loaded")
            except Exception as e:
                st.sidebar.error(f"Failed to load {name}: {str(e)}")
        else:
            st.sidebar.warning(f"{name} model not found")
    
    return models

# Load optimization results
@st.cache_data
def load_optimization_results():
    """Load sensor optimization results"""
    optimization_files = {
        'feature_importance': 'results/sensor_optimization_fixed/incremental_removal_results.csv',
        'sensor_combinations': 'results/sensor_optimization_fixed/sensor_combinations.csv',
        'recommendations': 'results/sensor_optimization_fixed/recommendations.csv'
    }
    
    results = {}
    for name, filepath in optimization_files.items():
        if os.path.exists(filepath):
            try:
                results[name] = pd.read_csv(filepath)
            except Exception as e:
                st.sidebar.warning(f"Could not load {name}: {e}")
    
    return results

# Load training data for optimization model
@st.cache_data
def load_training_data():
    """Load and prepare training data for optimization models"""
    # Try to load imputed data first, then raw data
    imputed_path = 'results/imputation_results/df_imputed.csv'
    raw_path = 'Machine Downtime.csv'
    
    if os.path.exists(imputed_path):
        df = pd.read_csv(imputed_path, parse_dates=['Date'], dayfirst=True)
    elif os.path.exists(raw_path):
        df = pd.read_csv(raw_path, parse_dates=['Date'], dayfirst=True)
    else:
        return None, None, None
    
    # Map machine IDs
    machine_map = {
        'Makino-L1-Unit1-2013': 'M1',
        'Makino-L2-Unit1-2015': 'M2',
        'Makino-L3-Unit1-2015': 'M3'
    }
    df['Machine_ID'] = df['Machine_ID'].map(machine_map).fillna(df['Machine_ID'])
    df['y'] = (df['Downtime'] == 'Machine_Failure').astype(int)
    
    # Get sensor columns
    exclude_cols = {'Machine_ID', 'Assembly_Line_No', 'Date', 'Downtime', 'y'}
    sensor_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
    
    X = df[sensor_cols].fillna(df[sensor_cols].median())
    y = df['y']
    
    return X, y, sensor_cols

# Get sensor importance ranking
@st.cache_data
def get_sensor_importance_ranking():
    """Get sensor ranking by importance from optimization results or training data"""
    
    # First try to use existing optimization results
    optimization_results = load_optimization_results()
    if optimization_results and 'feature_importance' in optimization_results:
        # Use incremental removal results to infer importance
        removal_df = optimization_results['feature_importance']
        if not removal_df.empty:
            # Sensors removed last are most important
            # Find the order sensors were removed by looking at which sensors remain
            all_sensors = list(SENSOR_DEFAULTS.keys())
            
            # Sort by number of sensors remaining (descending) to get removal order
            removal_df_sorted = removal_df.sort_values('n_sensors_remaining', ascending=False)
            
            # The first row has all sensors, subsequent rows show removal order
            sensor_importance_order = []
            prev_sensors = set(all_sensors)
            
            for _, row in removal_df_sorted.iterrows():
                if row['n_sensors_remaining'] == len(all_sensors):
                    continue  # Skip the "all sensors" row
                
                # Find which sensors were removed in this step
                current_sensors = prev_sensors - set([s.strip() for s in str(row.get('removed_sensors', '')).split(',') if s.strip()])
                removed_this_step = prev_sensors - current_sensors
                
                # Add removed sensors to the end (least important)
                for sensor in removed_this_step:
                    if sensor not in sensor_importance_order and sensor in all_sensors:
                        sensor_importance_order.append(sensor)
                
                prev_sensors = current_sensors
            
            # Add any remaining sensors to the beginning (most important)
            for sensor in all_sensors:
                if sensor not in sensor_importance_order:
                    sensor_importance_order.insert(0, sensor)
            
            # Reverse to get most important first
            return sensor_importance_order[::-1]
    
    # Fallback: Calculate from training data
    X, y, sensor_cols = load_training_data()
    
    if X is not None and len(X) > 100:  # Ensure we have enough data
        try:
            # Train a Random Forest to get feature importance
            rf = RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                class_weight='balanced',
                max_depth=10
            )
            rf.fit(X, y)
            
            # Get feature importance and rank sensors
            importance_df = pd.DataFrame({
                'sensor': sensor_cols,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)  # Most important first
            
            return importance_df['sensor'].tolist()
            
        except Exception as e:
            st.sidebar.warning(f"Could not calculate feature importance: {e}")
    
    # Final fallback: Domain knowledge based ranking
    # Based on typical CNC failure patterns
    return [
        'Spindle_Vibration(?m)',           # Vibration often first sign of mechanical issues
        'Hydraulic_Pressure(bar)',        # Critical for machine operation
        'Tool_Vibration(?m)',              # Tool wear indicator
        'Spindle_Bearing_Temperature(?C)', # Bearing failure common
        'Voltage(volts)',                  # Power stability important
        'Coolant_Temperature',             # Thermal management
        'Torque(Nm)',                      # Load indicator
        'Cutting(kN)',                     # Cutting force
        'Spindle_Speed(RPM)',              # Operating parameter
        'Hydraulic_Oil_Temperature(?C)',   # System temperature
        'Air_System_Pressure(bar)',        # Support systems
        'Coolant_Pressure(bar)'            # Coolant system
    ]

# Train optimized model with selected sensors
@st.cache_data
def train_optimized_model(selected_sensors, _X, _y):
    """Train a model using only selected sensors"""
    X_selected = _X[selected_sensors].copy()
    
    # Simple train-test split (using last machine as test set would be more realistic)
    # For demo purposes, using random split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, _y, test_size=0.3, stratify=_y, random_state=42
    )
    
    # Create and train pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Calculate performance metrics
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation score
    cv_scores = cross_val_score(pipeline, X_selected, _y, cv=3, scoring='roc_auc')
    cv_auc = cv_scores.mean()
    
    return pipeline, test_auc, cv_auc

# Load Model Performance Data
@st.cache_data
def load_model_performance():
    """Load model comparison results"""
    try:
        comparison_path = 'results/analysis/model_comparison.csv'
        if os.path.exists(comparison_path):
            df = pd.read_csv(comparison_path)
            return df
        else:
            return None
    except Exception as e:
        st.sidebar.error(f"Could not load model performance data: {e}")
        return None

# Load Model Metadata
@st.cache_data
def load_model_metadata():
    """Load model metadata to get expected features"""
    try:
        meta_path = 'results/analysis/model_meta.json'
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                return json.load(f)
        else:
            return None
    except Exception as e:
        st.sidebar.error(f"Could not load model metadata: {e}")
        return None

# Sensor Defaults and Ranges
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

# Sidebar: Model Selection and Info
st.sidebar.header("Model Selection")

# Load models and metadata
models = load_models()
model_performance = load_model_performance()
model_metadata = load_model_metadata()
optimization_results = load_optimization_results()

if not models:
    st.error("No models found! Please train models first by running the training script.")
    st.stop()

# Get expected features from metadata
expected_features = []
if model_metadata and 'sensors_used' in model_metadata:
    expected_features = model_metadata['sensors_used']
else:
    # Fallback to all sensor defaults if metadata not available
    expected_features = list(SENSOR_DEFAULTS.keys())

# Model selection
available_models = list(models.keys())
selected_model = st.sidebar.selectbox(
    "Choose Prediction Model:",
    available_models,
    help="Select which trained model to use for predictions"
)

# Display model performance if available
if model_performance is not None and selected_model:
    st.sidebar.header("Model Performance")
    model_perf = model_performance[model_performance['Model'].str.contains(selected_model.replace(' ', ''), case=False, na=False)]
    if model_perf.empty:
        model_perf = model_performance[model_performance['Model'] == selected_model]
    
    if not model_perf.empty:
        perf_data = model_perf.iloc[0]
        st.sidebar.metric("Test AUC", f"{perf_data['Test_AUC']:.3f}")
        st.sidebar.metric("Average Precision", f"{perf_data['Test_AP']:.3f}")
        st.sidebar.metric("CV AUC", f"{perf_data['CV_AUC_Mean']:.3f} ± {perf_data['CV_AUC_Std']:.3f}")
    else:
        st.sidebar.info("Performance data not available for selected model")

# Main Interface Modes
st.header("Select Operation Mode")

tab1, tab2, tab3, tab4 = st.tabs(["Failure Prediction", "Batch Prediction", "Model Analytics", "Optimized Prediction"])

# TAB 1: FAILURE PREDICTION (keeping existing code)
with tab1:
    st.subheader("Real-time Failure Prediction")
    
    st.info("Enter current sensor readings to predict machine failure probability")
    
    # Sensor input section
    st.subheader("Sensor Readings")
    
    # Create sensor input form in columns
    col1, col2, col3 = st.columns(3)
    
    sensor_data = {}
    
    # Group sensors logically
    pressure_sensors = [s for s in expected_features if 'Pressure' in s]
    temp_sensors = [s for s in expected_features if 'Temperature' in s]
    vibration_sensors = [s for s in expected_features if 'Vibration' in s]
    other_sensors = [s for s in expected_features if s not in pressure_sensors + temp_sensors + vibration_sensors]
    
    with col1:
        st.markdown("**Pressure Systems**")
        for sensor in pressure_sensors:
            min_val, max_val = SENSOR_RANGES.get(sensor, (0, 100))
            sensor_data[sensor] = st.number_input(
                sensor,
                min_value=float(min_val * 0.5),
                max_value=float(max_val * 1.5),
                value=float(SENSOR_DEFAULTS.get(sensor, 50.0)),
                step=0.1,
                key=f"pred_{sensor}"
            )
        
        st.markdown("**Temperature Systems**")
        for sensor in temp_sensors:
            min_val, max_val = SENSOR_RANGES.get(sensor, (0, 100))
            sensor_data[sensor] = st.number_input(
                sensor,
                min_value=float(min_val * 0.5),
                max_value=float(max_val * 1.5),
                value=float(SENSOR_DEFAULTS.get(sensor, 25.0)),
                step=0.1,
                key=f"pred_{sensor}"
            )
    
    with col2:
        st.markdown("**Vibration Systems**")
        for sensor in vibration_sensors:
            min_val, max_val = SENSOR_RANGES.get(sensor, (0, 10))
            sensor_data[sensor] = st.number_input(
                sensor,
                min_value=float(min_val * 0.2),
                max_value=float(max_val * 2.0),
                value=float(SENSOR_DEFAULTS.get(sensor, 1.0)),
                step=0.1,
                key=f"pred_{sensor}"
            )
    
    with col3:
        st.markdown("**Power & Performance**")
        for sensor in other_sensors:
            min_val, max_val = SENSOR_RANGES.get(sensor, (0, 1000))
            sensor_data[sensor] = st.number_input(
                sensor,
                min_value=float(min_val * 0.5),
                max_value=float(max_val * 1.5),
                value=float(SENSOR_DEFAULTS.get(sensor, 100.0)),
                step=0.1 if sensor != 'Spindle_Speed(RPM)' else 100.0,
                key=f"pred_{sensor}"
            )
    
    # Prediction button and results
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        predict_button = st.button("Predict Failure", type="primary", use_container_width=True)
    
    with col2:
        reset_button = st.button("Reset to Defaults", use_container_width=True)
    
    if reset_button:
        st.rerun()
    
    # ENHANCED PREDICTION HANDLING WITH BETTER ERROR CHECKING
    if predict_button:
        try:
            # Get the model
            model = models[selected_model]
            
            # Check if model is properly trained
            if hasattr(model, 'named_steps'):
                clf = model.named_steps.get('clf')
                if clf is None:
                    st.error(f"Model {selected_model} has no classifier component")
                    st.stop()
                
                # Check if the classifier is trained
                if not hasattr(clf, 'classes_') and not hasattr(clf, 'feature_importances_'):
                    st.error(f"Model {selected_model} appears to be untrained. Please retrain the models.")
                    st.info("Run Model.py to retrain all models")
                    st.stop()
            
            # Create input DataFrame with only expected features (no Machine_ID)
            df_input = pd.DataFrame([sensor_data])
            
            # Ensure all expected features are present
            for feature in expected_features:
                if feature not in df_input.columns:
                    df_input[feature] = SENSOR_DEFAULTS.get(feature, 0.0)
            
            # Reorder columns to match training data
            df_input = df_input[expected_features]
            
            # Validate input data
            if df_input.isnull().any().any():
                st.warning("Some input values are missing. Using defaults.")
                df_input = df_input.fillna(0)
            
            # Make prediction
            prediction = model.predict(df_input)[0]
            
            # Get probability with fallback
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(df_input)[0, 1]
            elif hasattr(model, 'decision_function'):
                # For models that don't have predict_proba
                score = model.decision_function(df_input)[0]
                probability = 1 / (1 + np.exp(-score))  # Sigmoid transformation
            else:
                probability = 0.5  # Fallback
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.markdown(
                        f'<div class="failure-warning"><h3>FAILURE PREDICTED</h3><p>Probability: {probability:.1%}</p></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="normal-status"><h3>NORMAL OPERATION</h3><p>Probability: {probability:.1%}</p></div>',
                        unsafe_allow_html=True
                    )
            
            with col2:
                # Risk level
                if probability >= 0.8:
                    risk_level = "Critical"
                    recommendation = "Stop machine immediately and perform maintenance"
                elif probability >= 0.6:
                    risk_level = "High"
                    recommendation = "Schedule maintenance within 24 hours"
                elif probability >= 0.3:
                    risk_level = "Medium"
                    recommendation = "Monitor closely and schedule maintenance"
                else:
                    risk_level = "Low"
                    recommendation = "Continue normal operation"
                
                st.metric("Risk Level", risk_level)
                st.info(f"**Recommendation:** {recommendation}")
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability,
                number={'valueformat': '.2%'},
                gauge={'axis': {'range': [0,1]},
                       'bar': {'thickness': 0.5},
                       'steps': [
                           {'range': [0, 0.3], 'color': '#d0e6f7'},
                           {'range': [0.3, 0.6], 'color': '#fed7aa'},
                           {'range': [0.6, 0.8], 'color': '#fdba74'},
                           {'range': [0.8, 1.0], 'color': '#fca5a5'}
                       ],
                       'threshold': {'line': {'color': 'red', 'width': 3}, 'thickness': 0.9, 'value': 0.5}}
            ))
            fig.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.error("Debug info: Check that model features match input features")
            
            # Enhanced debugging information
            st.subheader("Debug Information")
            st.write(f"**Selected Model:** {selected_model}")
            st.write(f"**Expected Features:** {len(expected_features)}")
            st.write(f"**Model Type:** {type(models.get(selected_model, 'Unknown'))}")
            
            if selected_model in models:
                model = models[selected_model]
                if hasattr(model, 'named_steps'):
                    clf = model.named_steps.get('clf')
                    if clf:
                        st.write(f"**Classifier Type:** {type(clf)}")
                        st.write(f"**Has classes_:** {hasattr(clf, 'classes_')}")
                        st.write(f"**Has feature_importances_:** {hasattr(clf, 'feature_importances_')}")
            
            st.info("Try retraining the models by running Model.py if the error persists.")

# TAB 2: BATCH PREDICTION (keeping existing code)
with tab2:
    st.subheader("Batch Prediction from CSV")
    st.caption("Upload a CSV file with sensor readings to predict failures for multiple data points")
    
    # Downloadable template
    template = pd.DataFrame(columns=expected_features)
    st.download_button(
        "Download Input Template CSV",
        data=template.to_csv(index=False),
        file_name="cnc_batch_template.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help=f"CSV should contain columns: {', '.join(expected_features)}"
    )
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! {len(data)} rows loaded.")
            st.dataframe(data.head(), use_container_width=True)
            
            missing_cols = [col for col in expected_features if col not in data.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                if st.button("Run Batch Prediction", type="primary"):
                    model = models[selected_model]
                    X = data[expected_features].copy()
                    
                    # Fill any missing values
                    for col in expected_features:
                        if col not in X.columns:
                            X[col] = SENSOR_DEFAULTS.get(col, 0.0)
                    
                    probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else np.zeros(len(X))
                    predictions = (probabilities >= 0.5).astype(int)
                    
                    # Results
                    results = data.copy()
                    results["Failure_Probability"] = probabilities
                    results["Predicted_Failure"] = predictions
                    results["Risk_Level"] = pd.cut(
                        probabilities, bins=[0, 0.3, 0.6, 0.8, 1.0],
                        labels=["Low", "Medium", "High", "Critical"], include_lowest=True
                    )
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Predictions", len(results))
                    with col2:
                        failure_count = int(results["Predicted_Failure"].sum())
                        st.metric("Predicted Failures", failure_count)
                    with col3:
                        failure_rate = failure_count / len(predictions) * 100
                        st.metric("Failure Rate", f"{failure_rate:.1f}%")
                    with col4:
                        avg_prob = results['Failure_Probability'].mean()
                        st.metric("Avg Failure Prob", f"{avg_prob:.1%}")
                    
                    # Visualization
                    fig = px.histogram(
                        results, x="Failure_Probability", nbins=20,
                        title="Distribution of Failure Probabilities",
                        color_discrete_sequence=['#1f77b4']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk breakdown
                    risk_counts = results['Risk_Level'].value_counts()
                    fig_pie = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Risk Level Distribution"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Results table
                    st.subheader("Detailed Results")
                    risk_filter = st.multiselect(
                        "Filter by Risk Level:",
                        options=['Low', 'Medium', 'High', 'Critical'],
                        default=['High', 'Critical']
                    )
                    
                    if risk_filter:
                        filtered_results = results[results['Risk_Level'].isin(risk_filter)]
                    else:
                        filtered_results = results
                    
                    st.dataframe(
                        filtered_results[['Predicted_Failure', 'Failure_Probability', 'Risk_Level']],
                        use_container_width=True
                    )
                    
                    # Download results
                    csv_results = results.to_csv(index=False)
                    st.download_button(
                        label="Download Results CSV",
                        data=csv_results,
                        file_name=f"cnc_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# TAB 3: MODEL ANALYTICS (keeping existing code)
with tab3:
    st.subheader("Model Performance Analytics")
    
    if model_performance is not None:
        # Detailed metrics table
        st.subheader("Model Metrics")
        
        display_df = model_performance.copy()
        display_df['Test_AUC'] = display_df['Test_AUC'].apply(lambda x: f"{x:.4f}")
        display_df['Test_AP'] = display_df['Test_AP'].apply(lambda x: f"{x:.4f}")
        display_df['CV_AUC_Mean'] = display_df['CV_AUC_Mean'].apply(lambda x: f"{x:.4f}")
        display_df['CV_AUC_Std'] = display_df['CV_AUC_Std'].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Model": "Model Name",
                "Test_AUC": "Test AUC",
                "Test_AP": "Average Precision",
                "CV_AUC_Mean": "CV AUC Mean",
                "CV_AUC_Std": "CV AUC Std"
            }
        )
        
        # Model ranking
        st.subheader("Model Rankings")
        ranked_models = model_performance.sort_values('Test_AUC', ascending=False)
        
        for i, (_, row) in enumerate(ranked_models.iterrows(), 1):
            if i == 1:
                st.success(f"**{i}. {row['Model']}** - AUC: {row['Test_AUC']:.4f}")
            elif i == 2:
                st.info(f"**{i}. {row['Model']}** - AUC: {row['Test_AUC']:.4f}")
            elif i == 3:
                st.warning(f"**{i}. {row['Model']}** - AUC: {row['Test_AUC']:.4f}")
            else:
                st.write(f"**{i}. {row['Model']}** - AUC: {row['Test_AUC']:.4f}")
    
    else:
        st.warning("Model performance data not available. Please run the training script first.")

# TAB 4: NEW OPTIMIZED PREDICTION
with tab4:
    st.subheader("Sensor-Optimized Prediction")
    st.info("Use fewer sensors while maintaining prediction accuracy. Based on sensor importance analysis.")
    
    # Load training data and sensor rankings
    X_train, y_train, available_sensors = load_training_data()
    sensor_rankings = get_sensor_importance_ranking()
    
    if X_train is None:
        st.error("Training data not available. Please ensure the dataset exists.")
        st.stop()
    
    # Sensor selection controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration")
        
        # Number of sensors to use
        n_sensors = st.slider(
            "Number of sensors to use:",
            min_value=1,
            max_value=len(sensor_rankings),
            value=6,
            help="Select how many of the most important sensors to use"
        )
        
        selected_sensors = sensor_rankings[:n_sensors]
        
        st.write(f"**Selected sensors (top {n_sensors}):**")
        for i, sensor in enumerate(selected_sensors, 1):
            st.write(f"{i}. {sensor}")
        
        # Show optimization recommendations if available
        if optimization_results and 'recommendations' in optimization_results:
            st.subheader("Optimization Insights")
            recommendations = optimization_results['recommendations']
            if not recommendations.empty:
                st.write("**Recommended configurations:**")
                for _, row in recommendations.iterrows():
                    st.write(f"• {row['criterion']}: {int(row['n_sensors_needed'])} sensors (AUC: {row['auc']:.3f})")
        
        # Train optimized model button
        train_optimized = st.button("Train Optimized Model", type="primary")
    
    with col2:
        st.subheader("Sensor Importance Ranking")
        
        # Create simple table instead of horizontal bar chart
        importance_data = pd.DataFrame({
            'Rank': range(1, len(sensor_rankings) + 1),
            'Sensor': sensor_rankings,
            'Selected': ['✓' if sensor in selected_sensors else '' for sensor in sensor_rankings]
        })
        
        st.dataframe(
            importance_data,
            use_container_width=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width=70),
                "Sensor": st.column_config.TextColumn("Sensor Name", width=300),
                "Selected": st.column_config.TextColumn("Selected", width=80)
            },
            hide_index=True,
            height=400
        )
    
    # Model training and comparison
    if train_optimized:
        with st.spinner("Training optimized model..."):
            try:
                # Train optimized model
                opt_model, opt_test_auc, opt_cv_auc = train_optimized_model(
                    selected_sensors, X_train, y_train
                )
                
                # Get baseline performance for comparison
                baseline_auc = 0.85  # Default baseline
                if model_performance is not None and selected_model:
                    model_perf = model_performance[
                        model_performance['Model'].str.contains(selected_model.replace(' ', ''), case=False, na=False)
                    ]
                    if not model_perf.empty:
                        baseline_auc = model_perf.iloc[0]['Test_AUC']
                
                # Calculate performance retention
                performance_retention = (opt_cv_auc / baseline_auc) * 100
                
                # Store optimized model in session state
                st.session_state['optimized_model'] = opt_model
                st.session_state['selected_sensors'] = selected_sensors
                st.session_state['opt_performance'] = {
                    'test_auc': opt_test_auc,
                    'cv_auc': opt_cv_auc,
                    'baseline_auc': baseline_auc,
                    'performance_retention': performance_retention
                }
                
                st.success(f"Optimized model trained successfully!")
                
                # Performance comparison
                st.subheader("Performance Comparison")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Sensors Used", f"{n_sensors}/12")
                with col2:
                    st.metric("CV AUC", f"{opt_cv_auc:.3f}")
                with col3:
                    st.metric("Baseline AUC", f"{baseline_auc:.3f}")
                with col4:
                    st.metric("Performance Retention", f"{performance_retention:.1f}%")
                
            except Exception as e:
                st.error(f"Failed to train optimized model: {str(e)}")
    
    # Prediction section with optimized model
    if 'optimized_model' in st.session_state:
        st.markdown("---")
        st.subheader("Make Predictions with Optimized Model")
        
        # Get stored model and sensors
        opt_model = st.session_state['optimized_model']
        selected_sensors = st.session_state['selected_sensors']
        opt_performance = st.session_state['opt_performance']
        
        # Sensor inputs for selected sensors only
        st.write(f"**Enter readings for the {len(selected_sensors)} most important sensors:**")
        
        # Create input form
        opt_sensor_data = {}
        n_cols = min(3, len(selected_sensors))
        cols = st.columns(n_cols)
        
        for i, sensor in enumerate(selected_sensors):
            col_idx = i % n_cols
            with cols[col_idx]:
                min_val, max_val = SENSOR_RANGES.get(sensor, (0, 100))
                opt_sensor_data[sensor] = st.number_input(
                    sensor,
                    min_value=float(min_val * 0.5),
                    max_value=float(max_val * 1.5),
                    value=float(SENSOR_DEFAULTS.get(sensor, 50.0)),
                    step=0.1,
                    key=f"opt_{sensor}"
                )
        
        # Prediction buttons
        col1, col2 = st.columns(2)
        with col1:
            opt_predict_button = st.button("Predict with Optimized Model", type="primary")
        with col2:
            compare_predict_button = st.button("Compare with Baseline Model")
        
        # Make predictions
        if opt_predict_button or compare_predict_button:
            try:
                # Prepare input for optimized model
                opt_df_input = pd.DataFrame([opt_sensor_data])
                opt_prediction = opt_model.predict(opt_df_input)[0]
                opt_probability = opt_model.predict_proba(opt_df_input)[0, 1]
                
                # Results display
                st.markdown("---")
                st.subheader("Optimized Model Results")
                
                if compare_predict_button:
                    # Also make prediction with baseline model
                    # Fill missing sensors with defaults for baseline
                    baseline_sensor_data = opt_sensor_data.copy()
                    for feature in expected_features:
                        if feature not in baseline_sensor_data:
                            baseline_sensor_data[feature] = SENSOR_DEFAULTS.get(feature, 0.0)
                    
                    baseline_df_input = pd.DataFrame([baseline_sensor_data])[expected_features]
                    baseline_model = models[selected_model]
                    baseline_prediction = baseline_model.predict(baseline_df_input)[0]
                    baseline_probability = baseline_model.predict_proba(baseline_df_input)[0, 1]
                    
                    # Side-by-side comparison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"### Optimized Model ({len(selected_sensors)} sensors)")
                        if opt_prediction == 1:
                            st.markdown(
                                f'<div class="failure-warning"><h4>FAILURE PREDICTED</h4><p>Probability: {opt_probability:.1%}</p></div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="normal-status"><h4>NORMAL OPERATION</h4><p>Probability: {opt_probability:.1%}</p></div>',
                                unsafe_allow_html=True
                            )
                    
                    with col2:
                        st.markdown(f"### Baseline Model (12 sensors)")
                        if baseline_prediction == 1:
                            st.markdown(
                                f'<div class="failure-warning"><h4>FAILURE PREDICTED</h4><p>Probability: {baseline_probability:.1%}</p></div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="normal-status"><h4>NORMAL OPERATION</h4><p>Probability: {baseline_probability:.1%}</p></div>',
                                unsafe_allow_html=True
                            )
                    
                    # Comparison metrics
                    st.subheader("Prediction Comparison")
                    comparison_metrics = pd.DataFrame({
                        'Model': [f'Optimized ({len(selected_sensors)} sensors)', 'Baseline (12 sensors)'],
                        'Prediction': ['Failure' if opt_prediction == 1 else 'Normal',
                                     'Failure' if baseline_prediction == 1 else 'Normal'],
                        'Probability': [f"{opt_probability:.1%}", f"{baseline_probability:.1%}"],
                        'Difference': [f"{abs(opt_probability - baseline_probability):.1%}", "Baseline"]
                    })
                    st.dataframe(comparison_metrics, use_container_width=True)
                    
                else:
                    # Single model results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if opt_prediction == 1:
                            st.markdown(
                                f'<div class="failure-warning"><h3>FAILURE PREDICTED</h3><p>Probability: {opt_probability:.1%}</p></div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="normal-status"><h3>NORMAL OPERATION</h3><p>Probability: {opt_probability:.1%}</p></div>',
                                unsafe_allow_html=True
                            )
                    
                    with col2:
                        # Confidence assessment
                        performance_retention = opt_performance['performance_retention']
                        if performance_retention >= 95:
                            confidence_class = "confidence-high"
                            confidence_text = "High Confidence"
                        elif performance_retention >= 90:
                            confidence_class = "confidence-medium" 
                            confidence_text = "Medium Confidence"
                        else:
                            confidence_class = "confidence-low"
                            confidence_text = "Lower Confidence"
                        
                        st.markdown(
                            f'<div class="{confidence_class}"><h4>{confidence_text}</h4><p>Model retains {performance_retention:.1f}% of baseline performance</p><p>Using {len(selected_sensors)}/{len(expected_features)} sensors</p></div>',
                            unsafe_allow_html=True
                        )
                
                # Gauge chart for optimized model
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=opt_probability,
                    number={'valueformat': '.2%'},
                    title={'text': f"Failure Probability (Optimized Model)"},
                    gauge={'axis': {'range': [0,1]},
                           'bar': {'thickness': 0.5, 'color': "darkblue"},
                           'steps': [
                               {'range': [0, 0.3], 'color': '#d0e6f7'},
                               {'range': [0.3, 0.6], 'color': '#fed7aa'},
                               {'range': [0.6, 0.8], 'color': '#fdba74'},
                               {'range': [0.8, 1.0], 'color': '#fca5a5'}
                           ],
                           'threshold': {'line': {'color': 'red', 'width': 3}, 'thickness': 0.9, 'value': 0.5}}
                ))
                fig.update_layout(height=300, margin=dict(l=10,r=10,t=50,b=10))
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    
    else:
        st.info("Please train an optimized model first to enable predictions.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        CNC Failure Prediction System | Built with Streamlit & Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)