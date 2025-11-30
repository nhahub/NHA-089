import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page Config
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load Model and Artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, scaler, model_columns

try:
    model, scaler, model_columns = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Please run train.py first.")
    st.stop()

# Title and Description
st.title("🎓 Student Score Prediction")
st.markdown("""
This app predicts the **Final Exam Score** of a student based on various factors like study hours, attendance, and more.
""")

# Input Form
with st.form("prediction_form"):
    st.subheader("Student Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hours_studied = st.number_input("Hours Studied", min_value=1, max_value=24, value=5)
        attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=80)
        sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
        previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=75)
        
    with col2:
        tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0, max_value=10, value=1)
        physical_activity = st.number_input("Physical Activity (hrs/week)", min_value=0, max_value=10, value=3)
        parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
        access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
        
    with col3:
        extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
        motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])
        family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
        
    col4, col5, col6 = st.columns(3)
    
    with col4:
        teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
        school_type = st.selectbox("School Type", ["Public", "Private"])
        
    with col5:
        peer_influence = st.selectbox("Peer Influence", ["Positive", "Negative", "Neutral"])
        learning_disabilities = st.selectbox("Learning Disabilities", ["Yes", "No"])
        
    with col6:
        parental_education = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
        distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
        gender = st.selectbox("Gender", ["Male", "Female"])

    submit_button = st.form_submit_button("Predict Score")

if submit_button:
    # Create DataFrame from inputs
    input_data = {
        'Hours_Studied': [hours_studied],
        'Attendance': [attendance],
        'Parental_Involvement': [parental_involvement],
        'Access_to_Resources': [access_to_resources],
        'Extracurricular_Activities': [extracurricular],
        'Sleep_Hours': [sleep_hours],
        'Previous_Scores': [previous_scores],
        'Motivation_Level': [motivation_level],
        'Internet_Access': [internet_access],
        'Tutoring_Sessions': [tutoring_sessions],
        'Family_Income': [family_income],
        'Teacher_Quality': [teacher_quality],
        'School_Type': [school_type],
        'Peer_Influence': [peer_influence],
        'Physical_Activity': [physical_activity],
        'Learning_Disabilities': [learning_disabilities],
        'Parental_Education_Level': [parental_education],
        'Distance_from_Home': [distance_from_home],
        'Gender': [gender]
    }
    
    df_input = pd.DataFrame(input_data)
    
    # Preprocessing
    # 1. Encoding
    df_encoded = pd.get_dummies(df_input, drop_first=True)
    
    # 2. Align columns with model
    # Get missing columns in the training test
    missing_cols = set(model_columns) - set(df_encoded.columns)
    # Add a missing column with default value 0
    for c in missing_cols:
        df_encoded[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    df_encoded = df_encoded[model_columns]
    
    # 3. Scaling
    numeric_cols = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Tutoring_Sessions']
    df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])
    
    # Prediction
    prediction_log = model.predict(df_encoded)
    prediction = np.expm1(prediction_log)[0]
    
    # Display Result
    st.success(f"Predicted Exam Score: **{prediction:.2f}**")
    
    # Simple Analysis
    if prediction >= 90:
        st.balloons()
        st.write("Excellent! Keep up the good work!")
    elif prediction >= 75:
        st.write("Great job! You're doing well.")
    elif prediction >= 60:
        st.write("Good effort, but there's room for improvement.")
    else:
        st.write("You might need to study a bit more. Don't give up!")
