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
from openai import OpenAI
from dotenv import load_dotenv
from utils import load_config, read_metrics_log

# Load environment variables from .env file
load_dotenv()


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
        # Try loading from current directory first
        if os.path.exists(path):
            return joblib.load(path)
        
        # If not found, try loading from 'Heart Attack Prediction' subdirectory
        # This handles the case when running from repository root (Streamlit Cloud)
        alt_path = os.path.join("Heart Attack Prediction", path)
        if os.path.exists(alt_path):
            return joblib.load(alt_path)
        
        # If still not found, raise an error
        raise FileNotFoundError(f"Model file not found at '{path}' or '{alt_path}'")
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

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.warning("⚠️ **Warning:** OPENAI_API_KEY not found in environment variables. Please create a .env file with your API key.")
    client = None
else:
    client = OpenAI(api_key=openai_api_key)


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


def preprocess_batch_data(df, feature_names):
    """Preprocess batch data for prediction, handling categorical values."""
    df_processed = df[feature_names].copy()
    
    # Process each column
    for col in feature_names:
        if col not in df_processed.columns:
            continue
            
        col_lower = col.lower()
        
        # Check if column contains string/categorical data
        if df_processed[col].dtype == 'object' or df_processed[col].dtype.name == 'category':
            # Handle gender/sex columns
            if 'gender' in col_lower or 'sex' in col_lower:
                # Convert Male/Female to 1/0 (case-insensitive)
                def convert_gender(x):
                    x_str = str(x).lower().strip()
                    if x_str.startswith('m'):
                        return 1.0
                    elif x_str.startswith('f'):
                        return 0.0
                    else:
                        # Default to 0 if unknown
                        return 0.0
                
                df_processed[col] = df_processed[col].apply(convert_gender)
            else:
                # Try to convert to numeric
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        else:
            # Already numeric, ensure it's float
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Fill any NaN values with 0
    df_processed = df_processed.fillna(0)
    
    # Convert to numpy array and ensure float type
    return df_processed.values.astype(float)


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
                # Store prediction results in session state
                st.session_state.prediction_result = {
                    'pred': pred,
                    'prob': prob,
                    'feature_values': x,
                    'feature_names': feature_names
                }
                
                # Reset chat history when new prediction is made
                st.session_state.chat_messages = []
                
                # Log prediction
                log_entry = {'timestamp': datetime.now().isoformat()}
                for i, fn in enumerate(feature_names):
                    log_entry[fn] = x[i]
                log_entry['prediction'] = pred
                log_entry['probability'] = prob if prob is not None else ''
                
                # Save to prediction logs
                log_df = pd.DataFrame([log_entry])
                if os.path.exists('prediction_logs.csv'):
                    log_df.to_csv('prediction_logs.csv', mode='a', header=False, index=False)
                else:
                    log_df.to_csv('prediction_logs.csv', index=False)
        
        # Display prediction results if they exist in session state
        if 'prediction_result' in st.session_state:
            pred = st.session_state.prediction_result['pred']
            prob = st.session_state.prediction_result['prob']
            
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
                
            st.success("✅ Prediction logged successfully!")
            
            # Chat Assistant Section
            st.markdown("---")
            st.title("🩺 Virtual Heart Health Assistant")
            st.markdown("### Get personalized advice and ask questions about your heart health")
            
            # Initialize chat history in session state and generate automatic advice
            if 'chat_messages' not in st.session_state or len(st.session_state.chat_messages) == 0:
                # Get patient data from session state
                feature_names = st.session_state.prediction_result.get('feature_names', [])
                feature_values = st.session_state.prediction_result.get('feature_values', [])
                
                # Convert to lists if they are numpy arrays
                if hasattr(feature_names, 'tolist'):
                    feature_names = feature_names.tolist()
                if hasattr(feature_values, 'tolist'):
                    feature_values = feature_values.tolist()
                
                # Format patient data for the assistant
                patient_data_str = ""
                if len(feature_names) > 0 and len(feature_values) > 0:
                    patient_data_str = "\n\nPatient Health Data:\n"
                    for i, (name, value) in enumerate(zip(feature_names, feature_values)):
                        # Convert numpy types to Python native types
                        if hasattr(value, 'item'):
                            value = value.item()
                        
                        # Format the value nicely
                        if isinstance(value, (int, float)):
                            if value == 0 or value == 1:
                                value_str = "Yes" if value == 1 else "No"
                            else:
                                value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
                        else:
                            value_str = str(value)
                        patient_data_str += f"- {name}: {value_str}\n"
                
                # Generate automatic personalized advice based on results
                risk_level = "HIGH RISK" if pred == 1 else "LOW RISK"
                risk_probability = f"{prob:.1%}" if prob is not None else "N/A"
                
                with st.spinner("🩺 Analyzing your results and preparing personalized advice..."):
                    try:
                        if client is None:
                            raise Exception("OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
                        
                        # Build system prompt for initial advice with patient data
                        system_prompt = f"""You are a compassionate and knowledgeable virtual heart health assistant. 
A patient has just received their heart attack risk prediction results:
- Risk Level: {risk_level}
- Risk Probability: {risk_probability}
- Prediction: {'Heart Attack Risk Detected' if pred == 1 else 'No Immediate Risk Detected'}
{patient_data_str}

Analyze the patient's health data and provide personalized, actionable advice based on their specific values and risk level. 
Be empathetic and supportive. Reference specific values from their data when relevant.
For HIGH RISK: Focus on immediate actions, lifestyle changes related to their specific health metrics, when to see a doctor, and preventive measures.
For LOW RISK: Focus on maintaining good health, preventive measures, and staying healthy based on their current values.
Always remind them to consult with their healthcare provider for medical decisions.
Keep the advice concise but comprehensive (3-4 paragraphs). End by asking if they have any questions."""
                        
                        # Get automatic advice from OpenAI
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": "Please provide me with personalized advice based on my heart attack risk assessment results and my health data."}
                            ],
                            temperature=0.7,
                            max_tokens=600
                        )
                        
                        automatic_advice = response.choices[0].message.content
                        
                        # Store the initial advice in chat history
                        st.session_state.chat_messages = [{
                            "role": "assistant",
                            "content": automatic_advice
                        }]
                        
                    except Exception as e:
                        # Check if it's a quota/API error
                        error_str = str(e)
                        is_quota_error = "429" in error_str or "insufficient_quota" in error_str.lower() or "quota" in error_str.lower()
                        
                        # Fallback message if OpenAI fails
                        risk_level = "HIGH RISK" if pred == 1 else "LOW RISK"
                        risk_probability = f"{prob:.1%}" if prob is not None else "N/A"
                        
                        # Generate personalized fallback advice using patient data
                        patient_insights = ""
                        if len(feature_names) > 0 and len(feature_values) > 0:
                            patient_insights = "\n\n**Based on your health data:**\n"
                            # Add specific insights based on common health metrics
                            for name, value in zip(feature_names, feature_values):
                                # Convert numpy types to Python native types
                                if hasattr(value, 'item'):
                                    value = value.item()
                                
                                name_lower = name.lower()
                                if 'age' in name_lower:
                                    if isinstance(value, (int, float)) and value > 50:
                                        patient_insights += f"- Your age ({value:.0f} years) is a factor in heart health. Regular monitoring becomes increasingly important.\n"
                                elif 'cholesterol' in name_lower or 'chol' in name_lower:
                                    if isinstance(value, (int, float)) and value > 200:
                                        patient_insights += f"- Your {name} level ({value:.1f}) may need attention. Consider discussing cholesterol management with your doctor.\n"
                                elif 'blood' in name_lower and 'pressure' in name_lower:
                                    if isinstance(value, (int, float)) and value > 120:
                                        patient_insights += f"- Your {name} ({value:.1f}) should be monitored regularly.\n"
                                elif 'smoking' in name_lower or 'smoke' in name_lower:
                                    if value == 1:
                                        patient_insights += f"- Quitting smoking is one of the most important steps you can take for heart health.\n"
                                elif 'exercise' in name_lower or 'activity' in name_lower:
                                    if value == 0:
                                        patient_insights += f"- Regular physical activity is crucial for heart health. Consider starting with light exercises.\n"
                        
                        if pred == 1:
                            fallback_advice = f"""Based on your heart attack risk assessment, you have been identified as **HIGH RISK** (Risk Probability: {risk_probability}).{patient_insights}

**Immediate Actions:**
- Please consult with a healthcare provider as soon as possible for a comprehensive evaluation
- If you experience chest pain, shortness of breath, or other concerning symptoms, seek emergency medical attention immediately

**Lifestyle Recommendations:**
- Follow a heart-healthy diet (reduce sodium, saturated fats, and processed foods)
- Engage in regular physical activity as recommended by your doctor
- Manage stress through relaxation techniques
- Avoid smoking and limit alcohol consumption
- Monitor your blood pressure and cholesterol levels regularly

**Important:** This is a risk assessment tool, not a medical diagnosis. Always consult with qualified healthcare professionals for medical advice and treatment decisions.

Do you have any questions about your heart health or these recommendations?"""
                        else:
                            fallback_advice = f"""Great news! Based on your heart attack risk assessment, you have been identified as **LOW RISK** (Risk Probability: {risk_probability}).{patient_insights}

**Maintaining Your Heart Health:**
- Continue following a balanced, heart-healthy diet rich in fruits, vegetables, whole grains, and lean proteins
- Stay physically active with regular exercise (at least 150 minutes of moderate activity per week)
- Maintain a healthy weight
- Get adequate sleep (7-9 hours per night)
- Manage stress effectively
- Avoid smoking and limit alcohol consumption

**Preventive Measures:**
- Continue regular health check-ups with your healthcare provider
- Monitor your blood pressure, cholesterol, and blood sugar levels
- Stay up to date with recommended health screenings

**Important:** While your current risk is low, maintaining these healthy habits is key to keeping your heart healthy long-term.

Do you have any questions about maintaining your heart health or preventive measures?"""
                        
                        # Show appropriate error message
                        if is_quota_error:
                            st.warning("⚠️ **Note:** The AI assistant service is currently unavailable due to API quota limits. However, I've provided personalized advice based on your results below.")
                        else:
                            st.warning("⚠️ **Note:** Unable to connect to AI assistant. However, I've provided personalized advice based on your results below.")
                        
                        st.session_state.chat_messages = [{
                            "role": "assistant",
                            "content": fallback_advice
                        }]
            
            # Display chat history
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask me anything about your heart health..."):
                # Add user message to chat history
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": prompt
                })
                
                # Generate assistant response
                with st.spinner("Thinking..."):
                    try:
                        # Get patient data from session state
                        feature_names = st.session_state.prediction_result.get('feature_names', [])
                        feature_values = st.session_state.prediction_result.get('feature_values', [])
                        
                        # Convert to lists if they are numpy arrays
                        if hasattr(feature_names, 'tolist'):
                            feature_names = feature_names.tolist()
                        if hasattr(feature_values, 'tolist'):
                            feature_values = feature_values.tolist()
                        
                        # Format patient data for the assistant
                        patient_data_str = ""
                        if len(feature_names) > 0 and len(feature_values) > 0:
                            patient_data_str = "\n\nPatient Health Data:\n"
                            for name, value in zip(feature_names, feature_values):
                                # Convert numpy types to Python native types
                                if hasattr(value, 'item'):
                                    value = value.item()
                                
                                # Format the value nicely
                                if isinstance(value, (int, float)):
                                    if value == 0 or value == 1:
                                        value_str = "Yes" if value == 1 else "No"
                                    else:
                                        value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
                                else:
                                    value_str = str(value)
                                patient_data_str += f"- {name}: {value_str}\n"
                        
                        # Build context for the assistant
                        risk_level = "HIGH RISK" if pred == 1 else "LOW RISK"
                        risk_probability = f"{prob:.1%}" if prob is not None else "N/A"
                        
                        system_prompt = f"""You are a compassionate and knowledgeable virtual heart health assistant. 
You are helping a patient who just received their heart attack risk prediction results:
- Risk Level: {risk_level}
- Risk Probability: {risk_probability}
- Prediction: {'Heart Attack Risk Detected' if pred == 1 else 'No Immediate Risk Detected'}
{patient_data_str}

Use the patient's health data to provide personalized, relevant advice. Reference specific values from their data when answering questions.
Provide helpful, accurate, and empathetic advice about heart health. Always remind patients to consult with their healthcare provider for medical decisions. 
Focus on lifestyle recommendations, preventive measures, and general heart health information based on their specific health metrics."""
                        
                        # Prepare messages for OpenAI
                        messages = [
                            {"role": "system", "content": system_prompt}
                        ]
                        
                        # Add chat history (last 10 messages to avoid token limits)
                        for msg in st.session_state.chat_messages[-10:]:
                            messages.append({
                                "role": msg["role"],
                                "content": msg["content"]
                            })
                        
                        # Get response from OpenAI
                        if client is None:
                            raise Exception("OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
                        
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=messages,
                            temperature=0.7,
                            max_tokens=500
                        )
                        
                        assistant_response = response.choices[0].message.content
                        
                        # Add assistant response to chat history
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": assistant_response
                        })
                        
                        # Rerun to display the new message
                        st.rerun()
                        
                    except Exception as e:
                        # Check if it's a quota/API error
                        error_str = str(e)
                        is_quota_error = "429" in error_str or "insufficient_quota" in error_str.lower() or "quota" in error_str.lower()
                        
                        if is_quota_error:
                            error_message = "I apologize, but the AI assistant service is currently unavailable due to API quota limits. Please try again later or consult with your healthcare provider for immediate questions."
                            st.warning("⚠️ **API Quota Exceeded:** The AI assistant service is temporarily unavailable. Please consult with your healthcare provider for medical advice.")
                        else:
                            error_message = f"I apologize, but I'm experiencing technical difficulties. Please try again later. Error: {str(e)}"
                            st.error("❌ **Error:** Unable to connect to AI assistant.")
                        
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": error_message
                        })
                        st.rerun()
            
            # Add a button to clear chat history
            if st.button("🔄 Start New Conversation", use_container_width=True):
                st.session_state.chat_messages = []
                
                # Get patient data from session state
                feature_names = st.session_state.prediction_result.get('feature_names', [])
                feature_values = st.session_state.prediction_result.get('feature_values', [])
                
                # Convert to lists if they are numpy arrays
                if hasattr(feature_names, 'tolist'):
                    feature_names = feature_names.tolist()
                if hasattr(feature_values, 'tolist'):
                    feature_values = feature_values.tolist()
                
                # Format patient data for the assistant
                patient_data_str = ""
                if len(feature_names) > 0 and len(feature_values) > 0:
                    patient_data_str = "\n\nPatient Health Data:\n"
                    for name, value in zip(feature_names, feature_values):
                        # Format the value nicely
                        if isinstance(value, (int, float)):
                            if value == 0 or value == 1:
                                value_str = "Yes" if value == 1 else "No"
                            else:
                                value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
                        else:
                            value_str = str(value)
                        patient_data_str += f"- {name}: {value_str}\n"
                
                risk_level = "HIGH RISK" if pred == 1 else "LOW RISK"
                risk_probability = f"{prob:.1%}" if prob is not None else "N/A"
                
                # Regenerate automatic advice
                try:
                    system_prompt = f"""You are a compassionate and knowledgeable virtual heart health assistant. 
A patient has just received their heart attack risk prediction results:
- Risk Level: {risk_level}
- Risk Probability: {risk_probability}
- Prediction: {'Heart Attack Risk Detected' if pred == 1 else 'No Immediate Risk Detected'}
{patient_data_str}

Analyze the patient's health data and provide personalized, actionable advice based on their specific values and risk level. 
Be empathetic and supportive. Reference specific values from their data when relevant.
For HIGH RISK: Focus on immediate actions, lifestyle changes related to their specific health metrics, when to see a doctor, and preventive measures.
For LOW RISK: Focus on maintaining good health, preventive measures, and staying healthy based on their current values.
Always remind them to consult with their healthcare provider for medical decisions.
Keep the advice concise but comprehensive (3-4 paragraphs). End by asking if they have any questions."""
                    
                    if client is None:
                        raise Exception("OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
                    
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": "Please provide me with personalized advice based on my heart attack risk assessment results and my health data."}
                        ],
                        temperature=0.7,
                        max_tokens=600
                    )
                    
                    automatic_advice = response.choices[0].message.content
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": automatic_advice
                    })
                except Exception as e:
                    # Check if it's a quota/API error
                    error_str = str(e)
                    is_quota_error = "429" in error_str or "insufficient_quota" in error_str.lower() or "quota" in error_str.lower()
                    
                    # Generate personalized fallback advice using patient data
                    patient_insights = ""
                    if len(feature_names) > 0 and len(feature_values) > 0:
                        patient_insights = "\n\n**Based on your health data:**\n"
                        # Add specific insights based on common health metrics
                        for name, value in zip(feature_names, feature_values):
                            # Convert numpy types to Python native types
                            if hasattr(value, 'item'):
                                value = value.item()
                            
                            name_lower = name.lower()
                            if 'age' in name_lower:
                                if isinstance(value, (int, float)) and value > 50:
                                    patient_insights += f"- Your age ({value:.0f} years) is a factor in heart health. Regular monitoring becomes increasingly important.\n"
                            elif 'cholesterol' in name_lower or 'chol' in name_lower:
                                if isinstance(value, (int, float)) and value > 200:
                                    patient_insights += f"- Your {name} level ({value:.1f}) may need attention. Consider discussing cholesterol management with your doctor.\n"
                            elif 'blood' in name_lower and 'pressure' in name_lower:
                                if isinstance(value, (int, float)) and value > 120:
                                    patient_insights += f"- Your {name} ({value:.1f}) should be monitored regularly.\n"
                            elif 'smoking' in name_lower or 'smoke' in name_lower:
                                if value == 1:
                                    patient_insights += f"- Quitting smoking is one of the most important steps you can take for heart health.\n"
                            elif 'exercise' in name_lower or 'activity' in name_lower:
                                if value == 0:
                                    patient_insights += f"- Regular physical activity is crucial for heart health. Consider starting with light exercises.\n"
                    
                    # Fallback message
                    if pred == 1:
                        fallback_advice = f"""Based on your heart attack risk assessment, you have been identified as **HIGH RISK** (Risk Probability: {risk_probability}).{patient_insights}

**Immediate Actions:**
- Please consult with a healthcare provider as soon as possible for a comprehensive evaluation
- If you experience chest pain, shortness of breath, or other concerning symptoms, seek emergency medical attention immediately

**Lifestyle Recommendations:**
- Follow a heart-healthy diet (reduce sodium, saturated fats, and processed foods)
- Engage in regular physical activity as recommended by your doctor
- Manage stress through relaxation techniques
- Avoid smoking and limit alcohol consumption
- Monitor your blood pressure and cholesterol levels regularly

**Important:** This is a risk assessment tool, not a medical diagnosis. Always consult with qualified healthcare professionals for medical advice and treatment decisions.

Do you have any questions about your heart health or these recommendations?"""
                    else:
                        fallback_advice = f"""Great news! Based on your heart attack risk assessment, you have been identified as **LOW RISK** (Risk Probability: {risk_probability}).{patient_insights}

**Maintaining Your Heart Health:**
- Continue following a balanced, heart-healthy diet rich in fruits, vegetables, whole grains, and lean proteins
- Stay physically active with regular exercise (at least 150 minutes of moderate activity per week)
- Maintain a healthy weight
- Get adequate sleep (7-9 hours per night)
- Manage stress effectively
- Avoid smoking and limit alcohol consumption

**Preventive Measures:**
- Continue regular health check-ups with your healthcare provider
- Monitor your blood pressure, cholesterol, and blood sugar levels
- Stay up to date with recommended health screenings

**Important:** While your current risk is low, maintaining these healthy habits is key to keeping your heart healthy long-term.

Do you have any questions about maintaining your heart health or preventive measures?"""
                    
                    # Show appropriate error message
                    if is_quota_error:
                        st.warning("⚠️ **Note:** The AI assistant service is currently unavailable due to API quota limits. However, I've provided personalized advice based on your results below.")
                    else:
                        st.warning("⚠️ **Note:** Unable to connect to AI assistant. However, I've provided personalized advice based on your results below.")
                    
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": fallback_advice
                    })
                    
                    st.rerun()

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
                            # Prepare data with preprocessing for categorical values
                            X = preprocess_batch_data(df, feature_names)
                            
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
        # Convert to list if it's a numpy array
        if hasattr(feature_names, 'tolist'):
            feature_names = feature_names.tolist()
        
        if len(feature_names) > 0:
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
