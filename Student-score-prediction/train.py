import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define paths
DATA_PATH = "StudentPerformanceFactors.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

def load_data():
    if os.path.exists(DATA_PATH):
        print(f"Loading data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
    else:
        print(f"Data file {DATA_PATH} not found. Generating synthetic data for demonstration...")
        # Generate synthetic data based on notebook analysis
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Hours_Studied': np.random.randint(1, 45, n_samples),
            'Attendance': np.random.randint(60, 100, n_samples),
            'Parental_Involvement': np.random.choice(['Low', 'Medium', 'High'], n_samples),
            'Access_to_Resources': np.random.choice(['Low', 'Medium', 'High'], n_samples),
            'Extracurricular_Activities': np.random.choice(['Yes', 'No'], n_samples),
            'Sleep_Hours': np.random.randint(4, 10, n_samples),
            'Previous_Scores': np.random.randint(50, 100, n_samples),
            'Motivation_Level': np.random.choice(['Low', 'Medium', 'High'], n_samples),
            'Internet_Access': np.random.choice(['Yes', 'No'], n_samples),
            'Tutoring_Sessions': np.random.randint(0, 8, n_samples),
            'Family_Income': np.random.choice(['Low', 'Medium', 'High'], n_samples),
            'Teacher_Quality': np.random.choice(['Low', 'Medium', 'High'], n_samples),
            'School_Type': np.random.choice(['Public', 'Private'], n_samples),
            'Peer_Influence': np.random.choice(['Positive', 'Negative', 'Neutral'], n_samples),
            'Physical_Activity': np.random.randint(0, 6, n_samples),
            'Learning_Disabilities': np.random.choice(['Yes', 'No'], n_samples),
            'Parental_Education_Level': np.random.choice(['High School', 'College', 'Postgraduate'], n_samples),
            'Distance_from_Home': np.random.choice(['Near', 'Moderate', 'Far'], n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples)
        }
        
        df = pd.DataFrame(data)
        # Generate target variable with some logic + noise
        df['Exam_Score'] = (
            50 + 
            0.5 * df['Hours_Studied'] + 
            0.3 * df['Attendance'] + 
            0.4 * df['Previous_Scores'] + 
            np.random.normal(0, 5, n_samples)
        )
        df['Exam_Score'] = df['Exam_Score'].clip(0, 100).astype(int)
        
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    
    # Handle missing values (if any, though synthetic won't have them)
    # Based on notebook:
    if 'Teacher_Quality' in df.columns:
        df['Teacher_Quality'] = df['Teacher_Quality'].fillna(df['Teacher_Quality'].mode()[0])
    if 'Parental_Education_Level' in df.columns:
        df['Parental_Education_Level'] = df['Parental_Education_Level'].fillna(df['Parental_Education_Level'].mode()[0])
    if 'Distance_from_Home' in df.columns:
        df['Distance_from_Home'] = df['Distance_from_Home'].fillna(df['Distance_from_Home'].mode()[0])
        
    # Drop rows with Exam_Score > 100 (from notebook)
    df = df[df['Exam_Score'] <= 100]
    
    # Encoding categorical variables
    # The notebook used get_dummies(drop_first=True)
    # We need to ensure we save the columns structure for inference
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Scaling numerical features
    scaler = StandardScaler()
    numeric_cols = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Tutoring_Sessions']
    
    # Ensure these columns exist
    for col in numeric_cols:
        if col not in df_encoded.columns:
             # If synthetic generation missed something or column name mismatch
             print(f"Warning: {col} not found in dataframe")
             
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
    
    return df_encoded, scaler, numeric_cols

def train_model():
    df = load_data()
    
    # Preprocess
    df_encoded, scaler, numeric_cols = preprocess_data(df)
    
    # Features & Target
    X = df_encoded.drop('Exam_Score', axis=1)
    y = df_encoded['Exam_Score']
    
    # Log transform target (from notebook)
    # Handle zeros if any
    y_log = np.log1p(y) # using log1p to be safe with 0s, though notebook used np.log
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    
    # Model Training
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log) # Inverse of log1p
    y_test_orig = np.expm1(y_test)
    
    mse = mean_squared_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)
    
    print(f"Model Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Save Artifacts
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    
    print(f"Saving scaler to {SCALER_PATH}...")
    joblib.dump(scaler, SCALER_PATH)
    
    # Save feature columns for inference alignment
    model_columns = list(X.columns)
    joblib.dump(model_columns, "model_columns.pkl")
    print("Saved model columns for inference.")
    
    print("Training complete.")

if __name__ == "__main__":
    train_model()
