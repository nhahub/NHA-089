# --- STYLES & COLORS ---
C_BG = '#f8fafc'
C_CARD = '#ffffff'
C_ACCENT = '#2563eb'
C_BORDER = '#e2e8f0'

S_MAIN = {'fontFamily': '"Inter", "Segoe UI", sans-serif', 'backgroundColor': C_BG, 'minHeight': '100vh', 'margin': 0, 'padding': 0}
S_HEADER = {'backgroundColor': '#1e293b', 'color': 'white', 'padding': '25px 20px', 'textAlign': 'center', 'marginBottom': '30px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'}
S_CONTAINER = {'maxWidth': '1300px', 'margin': '0 auto', 'padding': '0 20px', 'display': 'flex', 'flexWrap': 'wrap', 'gap': '25px'}
S_CARD = {'backgroundColor': C_CARD, 'borderRadius': '12px', 'padding': '25px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.04)', 'border': f'1px solid {C_BORDER}', 'marginBottom': '20px'}
S_GROUP_TITLE = {'fontSize': '0.9rem', 'fontWeight': '700', 'color': '#64748b', 'textTransform': 'uppercase', 'letterSpacing': '0.05em', 'borderBottom': f'2px solid {C_BG}', 'paddingBottom': '10px', 'marginBottom': '20px', 'marginTop': 0}
S_BTN = {'width': '100%', 'padding': '16px', 'backgroundColor': C_ACCENT, 'color': 'white', 'border': 'none', 'borderRadius': '8px', 'fontSize': '1.1rem', 'fontWeight': '700', 'cursor': 'pointer', 'transition': 'all 0.2s', 'boxShadow': '0 4px 10px rgba(37, 99, 235, 0.2)'}
S_LABEL = {'display': 'block', 'fontSize': '0.85rem', 'fontWeight': '600', 'color': '#475569', 'marginBottom': '6px'}

# --- FEATURE GROUPS ---
GROUPS = {
    "Patient Profile": ['gender', 'age', 'region', 'income_level'],
    "Vitals & Labs": ['blood_pressure_systolic', 'blood_pressure_diastolic', 'cholesterol_level', 'fasting_blood_sugar', 'triglycerides', 'cholesterol_hdl', 'cholesterol_ldl', 'waist_circumference', 'EKG_results'],
    "Lifestyle": ['smoking_status', 'physical_activity', 'dietary_habits', 'sleep_hours', 'air_pollution_exposure', 'stress_level', 'alcohol_consumption'],  # added alcohol
    "Medical History": ['family_history', 'hypertension', 'diabetes', 'obesity', 'previous_heart_disease', 'medication_usage', 'participated_in_free_screening']
}

NUMERIC_FEATURES = [
    'age', 'cholesterol_level', 'waist_circumference', 'sleep_hours', 
    'blood_pressure_systolic', 'blood_pressure_diastolic', 
    'fasting_blood_sugar', 'cholesterol_hdl', 'cholesterol_ldl', 
    'triglycerides'
]

STRING_CATEGORICALS = [
    'gender', 'region', 'income_level', 'smoking_status', 
    'physical_activity', 'dietary_habits', 'air_pollution_exposure', 
    'stress_level', 'EKG_results', 'alcohol_consumption'  # added alcohol
]

BINARY_NUMERIC_FEATURES = [
    'hypertension', 'diabetes', 'obesity', 'family_history', 
    'previous_heart_disease', 'medication_usage', 
    'participated_in_free_screening'
]

ALL_MODEL_FEATURES = []
for group in GROUPS.values():
    ALL_MODEL_FEATURES.extend(group)

HUMAN_READABLE_MAP = {
    'age': 'Age (Years)', 'gender': 'Gender', 'blood_pressure_systolic': 'Systolic BP',
    'blood_pressure_diastolic': 'Diastolic BP', 'fasting_blood_sugar': 'Blood Sugar',
    'cholesterol_hdl': 'HDL Cholesterol', 'cholesterol_ldl': 'LDL Cholesterol',
    'triglycerides': 'Triglycerides', 'smoking_status': 'Smoking Status',
    'physical_activity': 'Physical Activity', 'dietary_habits': 'Diet',
    'stress_level': 'Stress Level', 'sleep_hours': 'Sleep Hours',
    'waist_circumference': 'Waist Circ.', 'air_pollution_exposure': 'Pollution Exp.',
    'EKG_results': 'EKG Results', 'region': 'Region', 'income_level': 'Income Level',
    'previous_heart_disease': 'Prev. Heart Disease', 'medication_usage': 'Medication Usage',
    'participated_in_free_screening': 'Screening History', 'diabetes': 'Diabetes',
    'cholesterol_level': 'Total Cholesterol', 'hypertension': 'Hypertension',
    'family_history': 'Family History', 'obesity': 'Obesity', 'alcohol_consumption': 'Alcohol Consumption'
}

available_features_ui = {col: HUMAN_READABLE_MAP.get(col, col.replace('_', ' ').title()) for col in ALL_MODEL_FEATURES}