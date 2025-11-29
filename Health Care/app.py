"""
Enhanced Streamlit Application for Simplified MLOps System
Heart Attack Risk Prediction with Integrated Model Management
"""
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import json
import subprocess
import sys
from datetime import datetime
from utils import load_config, read_metrics_log


# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Heart Attack Risk Predictor - MLOps System",
    layout="wide",
    page_icon="❤️",
    initial_sidebar_state="expanded"
)


# ---------------------------
# Load Model and Configuration
# ---------------------------
@st.cache_resource
def load_model(path="heart_attack_final_model.pkl"):
    """Load the trained model."""
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def reload_model():
    """Force reload of the model."""
    st.cache_resource.clear()
    return load_model()


# Load model
model = load_model()

# Load configuration
config = load_config()


# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("🏥 MLOps System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🔮 Single Prediction", "📊 Batch Prediction", "📈 Monitoring Dashboard", "ℹ️ Model Information", "🔄 Retrain Model"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "**Simplified MLOps System**\n\n"
    "A lightweight alternative to full-scale MLOps, "
    "providing experiment tracking, model deployment, "
    "and monitoring without heavy tooling."
)


# ---------------------------
# Helper Functions
# ---------------------------
def get_feature_names():
    """Extract feature names from model."""
    if model is None:
        return []
    feature_names = getattr(model, 'feature_names_in_', None)
    if feature_names is None:
        n = getattr(model, 'n_features_in_', 10)
        feature_names = [f"feature_{i+1}" for i in range(n)]
    return feature_names


def smart_input(name, key_suffix=""):
    """Create appropriate input widget based on feature name."""
    n = name.lower()
    if 'gender' in n or 'sex' in n:
        return st.selectbox(f"{name}", ["Male", "Female", "Other"], key=f"{name}_{key_suffix}"), 'cat'
    elif 'history' in n or 'previous' in n or 'level' in n:
        return st.selectbox(f"{name}", [0, 1], key=f"{name}_{key_suffix}"), 'binary'
    else:
        return st.number_input(f"{name}", value=0.0, format="%f", key=f"{name}_{key_suffix}"), 'num'


def process_input(value, input_type):
    """Process input value based on type."""
    if input_type == 'cat':
        return float(1 if str(value).lower().startswith('m') else 0)
    else:
        return float(value)


def make_prediction(X):
    """Make prediction with the model."""
    if model is None:
        return None, None
    
    pred = model.predict(X)[0]
    prob = None
    
    if hasattr(model, 'predict_proba'):
        try:
            prob = float(model.predict_proba(X)[0][1])
        except:
            prob = None
    
    return pred, prob


# ---------------------------
# PAGE 1: Single Prediction
# ---------------------------
if page == "🔮 Single Prediction":
    st.title("❤️ Heart Attack Risk Predictor")
    st.markdown("### Single Patient Prediction")
    st.markdown("Enter patient information to predict heart attack risk.")
    
    if model is None:
        st.error("❌ Model not loaded. Please train the model first.")
    else:
        feature_names = get_feature_names()
        
        # Create form for inputs
        with st.form("prediction_form"):
            st.subheader("Patient Information")
            
            # Create columns for better layout
            cols = st.columns(2)
            inputs = {}
            types = {}
            
            for i, fn in enumerate(feature_names):
                col_idx = i % 2
                with cols[col_idx]:
                    widget, wtype = smart_input(fn, "single")
                    inputs[fn] = widget
                    types[fn] = wtype
            
            submit_button = st.form_submit_button("🔮 Predict Risk", use_container_width=True)
        
        # Process prediction
        if submit_button:
            x = [process_input(inputs[fn], types[fn]) for fn in feature_names]
            X = np.array([x])
            
            pred, prob = make_prediction(X)
            
            if pred is not None:
                st.markdown("---")
                st.subheader("📋 Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if pred == 1:
                        st.error(f"### ⚠️ High Risk")
                        st.markdown("**Prediction:** Heart Attack Risk Detected")
                    else:
                        st.success(f"### ✅ Low Risk")
                        st.markdown("**Prediction:** No Immediate Risk")
                
                with col2:
                    if prob is not None:
                        st.metric("Risk Probability", f"{prob:.2%}")
                        st.progress(prob)
                
                # Log prediction
                log_entry = {'timestamp': datetime.now().isoformat()}
                for i, fn in enumerate(feature_names):
                    log_entry[fn] = x[i]
                log_entry['prediction'] = pred
                log_entry['probability'] = prob if prob is not None else ''
                
                if os.path.exists('prediction_logs.csv'):
                    pd.DataFrame([log_entry]).to_csv('prediction_logs.csv', mode='a', header=False, index=False)
                else:
                    pd.DataFrame([log_entry]).to_csv('prediction_logs.csv', index=False)
                
                st.success("✅ Prediction logged successfully")


# ---------------------------
# PAGE 2: Batch Prediction
# ---------------------------
elif page == "📊 Batch Prediction":
    st.title("📊 Batch Prediction")
    st.markdown("### Upload CSV for Multiple Predictions")
    
    if model is None:
        st.error("❌ Model not loaded. Please train the model first.")
    else:
        feature_names = get_feature_names()
        
        st.info(f"📝 **Required columns:** {', '.join(feature_names)}")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with the same features as the training data"
        )
        
        if uploaded_file is not None:
            try:
                # Load uploaded data
                df = pd.read_csv(uploaded_file)
                
                st.subheader("📄 Uploaded Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                st.info(f"Total rows: {len(df)}")
                
                # Validate columns
                missing_cols = set(feature_names) - set(df.columns)
                if missing_cols:
                    st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
                else:
                    if st.button("🔮 Generate Predictions", type="primary"):
                        with st.spinner("Generating predictions..."):
                            # Prepare data
                            X = df[feature_names].values
                            
                            # Make predictions
                            predictions = model.predict(X)
                            
                            # Add predictions to dataframe
                            df_results = df.copy()
                            df_results['prediction'] = predictions
                            
                            if hasattr(model, 'predict_proba'):
                                try:
                                    probabilities = model.predict_proba(X)[:, 1]
                                    df_results['risk_probability'] = probabilities
                                except:
                                    pass
                            
                            st.success("✅ Predictions completed!")
                            
                            # Display results
                            st.subheader("📊 Prediction Results")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Predictions", len(predictions))
                            with col2:
                                high_risk = (predictions == 1).sum()
                                st.metric("High Risk Cases", high_risk)
                            with col3:
                                low_risk = (predictions == 0).sum()
                                st.metric("Low Risk Cases", low_risk)
                            
                            # Show results table
                            st.dataframe(df_results, use_container_width=True)
                            
                            # Download button
                            csv = df_results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")


# ---------------------------
# PAGE 3: Monitoring Dashboard
# ---------------------------
elif page == "📈 Monitoring Dashboard":
    st.title("📈 Monitoring Dashboard")
    st.markdown("### Model Performance and System Metrics")
    
    # Load metrics log
    metrics_df = read_metrics_log()
    
    if metrics_df is not None and len(metrics_df) > 0:
        # Summary metrics
        latest = metrics_df.iloc[-1]
        
        st.subheader("🎯 Current Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{latest['accuracy']:.4f}")
        with col2:
            st.metric("F1 Score", f"{latest['f1_score']:.4f}")
        with col3:
            st.metric("Precision", f"{latest['precision']:.4f}")
        with col4:
            st.metric("Recall", f"{latest['recall']:.4f}")
        
        st.markdown("---")
        
        # Historical trends
        st.subheader("📊 Historical Performance Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Accuracy Over Time")
            chart_data = metrics_df[['timestamp', 'accuracy']].copy()
            chart_data['timestamp'] = pd.to_datetime(chart_data['timestamp'])
            chart_data = chart_data.set_index('timestamp')
            st.line_chart(chart_data)
        
        with col2:
            st.markdown("#### F1 Score Over Time")
            chart_data = metrics_df[['timestamp', 'f1_score']].copy()
            chart_data['timestamp'] = pd.to_datetime(chart_data['timestamp'])
            chart_data = chart_data.set_index('timestamp')
            st.line_chart(chart_data)
        
        # Training metrics
        st.markdown("---")
        st.subheader("⚙️ Training Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Dataset Size Over Time")
            chart_data = metrics_df[['timestamp', 'dataset_size']].copy()
            chart_data['timestamp'] = pd.to_datetime(chart_data['timestamp'])
            chart_data = chart_data.set_index('timestamp')
            st.line_chart(chart_data)
        
        with col2:
            st.markdown("#### Training Duration Over Time")
            chart_data = metrics_df[['timestamp', 'training_duration']].copy()
            chart_data['timestamp'] = pd.to_datetime(chart_data['timestamp'])
            chart_data = chart_data.set_index('timestamp')
            st.line_chart(chart_data)
        
        # Full metrics table
        st.markdown("---")
        st.subheader("📋 Complete Metrics History")
        st.dataframe(metrics_df, use_container_width=True)
        
    else:
        st.info("📊 No metrics available yet. Train the model to start logging metrics.")
    
    # Prediction logs
    st.markdown("---")
    st.subheader("📝 Recent Predictions")
    
    if os.path.exists('prediction_logs.csv'):
        pred_df = pd.read_csv('prediction_logs.csv')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Predictions", len(pred_df))
        with col2:
            if 'prediction' in pred_df.columns:
                high_risk_pct = (pred_df['prediction'] == 1).mean() * 100
                st.metric("High Risk %", f"{high_risk_pct:.1f}%")
        
        st.dataframe(pred_df.tail(20), use_container_width=True)
        
        if 'probability' in pred_df.columns:
            st.markdown("#### Risk Probability Distribution")
            st.line_chart(pred_df['probability'].tail(100).fillna(0))
    else:
        st.info("📝 No prediction logs yet. Make predictions to start logging.")


# ---------------------------
# PAGE 4: Model Information
# ---------------------------
elif page == "ℹ️ Model Information":
    st.title("ℹ️ Model Information")
    st.markdown("### Current Model Configuration and Metadata")
    
    if model is None:
        st.warning("⚠️ No model loaded. Please train the model first.")
    else:
        # Model basics
        st.subheader("🤖 Model Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Type", config['model']['type'])
        with col2:
            st.metric("Version", config['model']['version'])
        with col3:
            last_trained = config['training'].get('last_trained', 'Never')
            st.metric("Last Trained", last_trained if last_trained else 'Never')
        
        # Hyperparameters
        st.markdown("---")
        st.subheader("⚙️ Hyperparameters")
        
        hyperparams_df = pd.DataFrame([config['model']['hyperparameters']]).T
        hyperparams_df.columns = ['Value']
        st.table(hyperparams_df)
        
        # Performance metrics
        st.markdown("---")
        st.subheader("📊 Current Performance Metrics")
        
        if config['metrics']['accuracy'] is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{config['metrics']['accuracy']:.4f}")
            with col2:
                st.metric("F1 Score", f"{config['metrics']['f1_score']:.4f}")
            with col3:
                st.metric("Precision", f"{config['metrics']['precision']:.4f}")
            with col4:
                st.metric("Recall", f"{config['metrics']['recall']:.4f}")
        else:
            st.info("📊 No performance metrics available. Train the model to generate metrics.")
        
        # Training configuration
        st.markdown("---")
        st.subheader("🎓 Training Configuration")
        
        training_config = {
            'Dataset Path': config['training']['dataset_path'],
            'Target Column': config['training']['target_column'],
            'Test Size': f"{config['training']['test_size'] * 100}%",
            'Dataset Size': config['training'].get('dataset_size', 'N/A'),
            'Training Duration': f"{config['training'].get('training_duration_seconds', 'N/A')} seconds"
        }
        
        training_df = pd.DataFrame([training_config]).T
        training_df.columns = ['Value']
        st.table(training_df)
        
        # Features
        st.markdown("---")
        st.subheader("📋 Model Features")
        
        feature_names = get_feature_names()
        if feature_names:
            st.info(f"**Number of features:** {len(feature_names)}")
            
            # Display in columns
            n_cols = 3
            cols = st.columns(n_cols)
            for i, feat in enumerate(feature_names):
                col_idx = i % n_cols
                with cols[col_idx]:
                    st.markdown(f"- {feat}")
        
        # Notes
        if config.get('notes'):
            st.markdown("---")
            st.subheader("📝 Notes")
            st.info(config['notes'])
        
        # Full configuration
        st.markdown("---")
        st.subheader("🔧 Full Configuration")
        
        with st.expander("View Raw Configuration"):
            st.json(config)


# ---------------------------
# PAGE 5: Retrain Model
# ---------------------------
elif page == "🔄 Retrain Model":
    st.title("🔄 Retrain Model")
    st.markdown("### Trigger Model Retraining")
    
    st.info(
        "🎯 **Retraining Process**\n\n"
        "This will:\n"
        "- Load the latest dataset\n"
        "- Train a new model with current hyperparameters\n"
        "- Evaluate performance\n"
        "- Update config.json and metrics_log.csv\n"
        "- Reload the model in this application"
    )
    
    # Display current config
    st.subheader("📋 Current Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dataset:**")
        st.code(config['training']['dataset_path'])
        
        st.markdown("**Model Type:**")
        st.code(config['model']['type'])
    
    with col2:
        st.markdown("**Current Version:**")
        st.code(config['model']['version'])
        
        st.markdown("**Last Trained:**")
        last_trained = config['training'].get('last_trained', 'Never')
        st.code(last_trained if last_trained else 'Never')
    
    # Hyperparameters editor
    st.markdown("---")
    st.subheader("⚙️ Adjust Hyperparameters (Optional)")
    
    with st.expander("Edit Hyperparameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.number_input(
                "Number of Estimators",
                min_value=10,
                max_value=500,
                value=config['model']['hyperparameters']['n_estimators'],
                step=10
            )
            
            max_depth = st.number_input(
                "Max Depth",
                min_value=1,
                max_value=50,
                value=config['model']['hyperparameters']['max_depth'],
                step=1
            )
        
        with col2:
            min_samples_split = st.number_input(
                "Min Samples Split",
                min_value=2,
                max_value=20,
                value=config['model']['hyperparameters']['min_samples_split'],
                step=1
            )
            
            min_samples_leaf = st.number_input(
                "Min Samples Leaf",
                min_value=1,
                max_value=20,
                value=config['model']['hyperparameters']['min_samples_leaf'],
                step=1
            )
        
        if st.button("💾 Save Hyperparameters"):
            config['model']['hyperparameters']['n_estimators'] = n_estimators
            config['model']['hyperparameters']['max_depth'] = max_depth
            config['model']['hyperparameters']['min_samples_split'] = min_samples_split
            config['model']['hyperparameters']['min_samples_leaf'] = min_samples_leaf
            
            from utils import save_config
            save_config(config)
            st.success("✅ Hyperparameters saved!")
    
    # Retrain button
    st.markdown("---")
    st.subheader("🚀 Start Retraining")
    
    if st.button("🔄 Retrain Model Now", type="primary", use_container_width=True):
        with st.spinner("🔄 Training in progress... This may take a few minutes."):
            try:
                # Run training script
                result = subprocess.run(
                    [sys.executable, "train.py"],
                    capture_output=True,
                    text=True,
                    cwd=os.getcwd()
                )
                
                if result.returncode == 0:
                    st.success("✅ Model retrained successfully!")
                    
                    # Show output
                    with st.expander("📋 Training Log"):
                        st.code(result.stdout)
                    
                    # Reload model and config
                    st.info("🔄 Reloading model and configuration...")
                    reload_model()
                    config = load_config()
                    
                    # Display new metrics
                    st.balloons()
                    st.subheader("📊 New Model Performance")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Accuracy", f"{config['metrics']['accuracy']:.4f}")
                    with col2:
                        st.metric("F1 Score", f"{config['metrics']['f1_score']:.4f}")
                    with col3:
                        st.metric("Precision", f"{config['metrics']['precision']:.4f}")
                    with col4:
                        st.metric("Recall", f"{config['metrics']['recall']:.4f}")
                    
                    st.info(f"🎯 New model version: **{config['model']['version']}**")
                    
                else:
                    st.error("❌ Training failed!")
                    
                    with st.expander("📋 Error Log"):
                        st.code(result.stderr)
                        st.code(result.stdout)
                    
            except Exception as e:
                st.error(f"❌ Error during retraining: {e}")


# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "❤️ Heart Attack Risk Predictor | Simplified MLOps System | "
    f"Model Version: {config['model']['version']}"
    "</div>",
    unsafe_allow_html=True
)