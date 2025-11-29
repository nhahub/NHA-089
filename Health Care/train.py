"""
Model training script for simplified MLOps system
"""
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from utils import (
    load_config, 
    save_config, 
    log_metrics, 
    update_config_after_training,
    validate_dataset
)


def preprocess_data(df: pd.DataFrame, target_column: str):
    """
    Preprocess dataset for training.
    
    Args:
        df: Raw dataset
        target_column: Name of target column
        
    Returns:
        Tuple of (X, y, feature_names)
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle categorical variables
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Fill missing values with median for numeric columns
    for col in X.columns:
        if X[col].isna().any():
            X[col].fillna(X[col].median(), inplace=True)
    
    feature_names = X.columns.tolist()
    
    return X, y, feature_names


def train_model(config_path: str = "config.json", model_path: str = "heart_attack_final_model.pkl"):
    """
    Train the model using configuration and dataset.
    
    Args:
        config_path: Path to configuration file
        model_path: Path to save trained model
        
    Returns:
        Dictionary with training results
    """
    print("=" * 60)
    print("Starting Model Training")
    print("=" * 60)
    
    # Load configuration
    config = load_config(config_path)
    print(f"\n✓ Configuration loaded from {config_path}")
    
    # Load dataset
    dataset_path = config['training']['dataset_path']
    target_column = config['training']['target_column']
    
    print(f"✓ Loading dataset from {dataset_path}...")
    try:
        df = pd.read_csv(dataset_path)
        print(f"  Dataset shape: {df.shape}")
    except FileNotFoundError:
        print(f"✗ Error: Dataset not found at {dataset_path}")
        return None
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None
    
    # Validate dataset
    is_valid, error_msg = validate_dataset(df, target_column)
    if not is_valid:
        print(f"✗ Dataset validation failed: {error_msg}")
        return None
    print(f"✓ Dataset validated successfully")
    
    # Start timing
    start_time = time.time()
    
    # Preprocess data
    print("\n✓ Preprocessing data...")
    X, y, feature_names = preprocess_data(df, target_column)
    print(f"  Features: {len(feature_names)}")
    print(f"  Samples: {len(X)}")
    
    # Split data
    test_size = config['training']['test_size']
    random_state = config['model']['hyperparameters']['random_state']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"✓ Data split: {len(X_train)} train, {len(X_test)} test")
    
    # Train model
    print("\n✓ Training model...")
    hyperparams = config['model']['hyperparameters']
    model = RandomForestClassifier(
        n_estimators=hyperparams['n_estimators'],
        max_depth=hyperparams['max_depth'],
        random_state=hyperparams['random_state'],
        min_samples_split=hyperparams['min_samples_split'],
        min_samples_leaf=hyperparams['min_samples_leaf'],
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print(f"  Model trained successfully")
    
    # Evaluate model
    print("\n✓ Evaluating model...")
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted')
    }
    
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    
    # Calculate training duration
    training_duration = time.time() - start_time
    print(f"\n✓ Training duration: {training_duration:.2f} seconds")
    
    # Save model
    joblib.dump(model, model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Update configuration
    config = update_config_after_training(
        config=config,
        metrics=metrics,
        training_duration=training_duration,
        dataset_size=len(df),
        features=feature_names,
        notes=f"Training completed at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    save_config(config, config_path)
    print(f"✓ Configuration updated in {config_path}")
    
    # Log metrics
    log_metrics(
        accuracy=metrics['accuracy'],
        f1_score=metrics['f1_score'],
        precision=metrics['precision'],
        recall=metrics['recall'],
        dataset_size=len(df),
        training_duration=training_duration,
        model_version=config['model']['version'],
        notes="Model training completed"
    )
    print(f"✓ Metrics logged to metrics_log.csv")
    
    print("\n" + "=" * 60)
    print("Training Completed Successfully!")
    print("=" * 60)
    
    return {
        'metrics': metrics,
        'training_duration': training_duration,
        'model_version': config['model']['version'],
        'dataset_size': len(df),
        'feature_count': len(feature_names)
    }


if __name__ == "__main__":
    results = train_model()
    if results:
        print("\n📊 Training Summary:")
        print(f"   Model Version: {results['model_version']}")
        print(f"   Dataset Size: {results['dataset_size']:,} samples")
        print(f"   Features: {results['feature_count']}")
        print(f"   Accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"   F1 Score: {results['metrics']['f1_score']:.4f}")
        print(f"   Duration: {results['training_duration']:.2f}s")
    else:
        print("\n✗ Training failed. Please check the errors above.")
