# src/train_with_mlflow.py (Standalone script)
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
from src.constants import ALL_MODEL_FEATURES
from preprocessing import apply_preprocessing, preprocessor
from sklearn.pipeline import Pipeline


mlflow.sklearn.autolog(log_input_examples=True, log_models=True, log_datasets=False)

data = pd.read_csv('data/heart_attack_prediction_indonesia.csv')

y = data.pop('heart_attack')

X = data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test = apply_preprocessing(X_train, X_test)

# Calculate required metrics
def log_relevant_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    

mlflow.set_experiment("Heart_attack_Risk_Prediction")

with mlflow.start_run(run_name="RandomForest_V0.1") as parent_run:

    with mlflow.start_run(run_name="random_forrest_run_1", nested=True):
    
        # Set intial parameters
        params = {
        "n_estimators": 50,
        "max_depth": 10,
        }

        # Log Params
        #mlflow.log_param("n_estimators", params["n_estimators"])
        #mlflow.log_param("max_depth", params["max_depth"])

        # Train Model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        print('Fitting completed!')

        # Evaluate model
        y_pred = model.predict(X_test)
        print('Prediction completed!')
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy}')

        # Save the final model for the dashboard's use
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

os.makedirs('models', exist_ok=True)
joblib.dump(model_pipeline, 'models/full_model_pipeline.joblib')

print("Full pipeline saved to 'models/full_model_pipeline.joblib'")