import pandas as pd
import joblib
import os
from .constants import ALL_MODEL_FEATURES, NUMERIC_FEATURES, available_features_ui, STRING_CATEGORICALS, BINARY_NUMERIC_FEATURES

class DataManager:
    def __init__(self):
        self.defaults = {}
        self.feature_importance_df = None
        self.model = None
        self._load_data()
        self._load_model()

    def _load_data(self):
        data_path = 'data/heart_attack_prediction_indonesia.csv'
        if os.path.exists(data_path):
            try:
                df = pd.read_csv(data_path)
                # df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=False).str.replace('[^A-Za-z0-9_]+', '', regex=True)
                
                # Defaults
                for col in ALL_MODEL_FEATURES:
                    if col in df.columns:
                        if col in NUMERIC_FEATURES:
                            self.defaults[col] = df[col].mean()
                        else:
                            if not df[col].empty: self.defaults[col] = df[col].mode()[0]
            except Exception as e:
                print(f"Error loading CSV: {e}")

    def _load_model(self):
        try:
            model_path = 'models/full_model_pipeline.joblib'
            self.model = joblib.load(model_path)
            print("Model Loaded.")
            
            # Extract feature importance from the classifier in the pipeline
            if hasattr(self.model, 'named_steps') and 'classifier' in self.model.named_steps:
                classifier = self.model.named_steps['classifier']
                if hasattr(classifier, 'feature_importances_'):
                    importances = classifier.feature_importances_
                    self.feature_importance_df = pd.DataFrame({
                        'feature': ALL_MODEL_FEATURES,
                        'importance': importances
                    }).sort_values('importance', ascending=True)
                    print(f"Feature importance loaded: {len(self.feature_importance_df)} features")
        except Exception as e:
            print(f"Error loading model: {e}")

data_manager = DataManager()