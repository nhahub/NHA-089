"""
Utility functions for simplified MLOps system
"""
import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        return get_default_config()
    
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config: Dict[str, Any], config_path: str = "config.json") -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to configuration file
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration structure.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "model": {
            "type": "RandomForestClassifier",
            "version": "1.0.0",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            }
        },
        "training": {
            "dataset_path": "heart_attack_prediction_indonesia.csv",
            "target_column": "target",
            "test_size": 0.2,
            "last_trained": None,
            "training_duration_seconds": None,
            "dataset_size": None
        },
        "metrics": {
            "accuracy": None,
            "f1_score": None,
            "precision": None,
            "recall": None
        },
        "features": [],
        "notes": "Default configuration"
    }


def log_metrics(
    accuracy: float,
    f1_score: float,
    precision: float,
    recall: float,
    dataset_size: int,
    training_duration: float,
    model_version: str,
    notes: str = "",
    log_path: str = "metrics_log.csv"
) -> None:
    """
    Log training metrics to CSV file.
    
    Args:
        accuracy: Model accuracy
        f1_score: Model F1 score
        precision: Model precision
        recall: Model recall
        dataset_size: Number of samples in dataset
        training_duration: Training duration in seconds
        model_version: Model version identifier
        notes: Optional notes
        log_path: Path to metrics log file
    """
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'accuracy': round(accuracy, 4),
        'f1_score': round(f1_score, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'dataset_size': dataset_size,
        'training_duration': round(training_duration, 2),
        'model_version': model_version,
        'notes': notes
    }
    
    df = pd.DataFrame([log_entry])
    
    if os.path.exists(log_path):
        df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df.to_csv(log_path, index=False)


def read_metrics_log(log_path: str = "metrics_log.csv") -> Optional[pd.DataFrame]:
    """
    Read metrics log from CSV file.
    
    Args:
        log_path: Path to metrics log file
        
    Returns:
        DataFrame with metrics history or None if file doesn't exist
    """
    if not os.path.exists(log_path):
        return None
    
    return pd.read_csv(log_path)


def update_config_after_training(
    config: Dict[str, Any],
    metrics: Dict[str, float],
    training_duration: float,
    dataset_size: int,
    features: list,
    notes: str = ""
) -> Dict[str, Any]:
    """
    Update configuration with training results.
    
    Args:
        config: Existing configuration
        metrics: Dictionary with accuracy, f1_score, precision, recall
        training_duration: Training duration in seconds
        dataset_size: Number of samples in dataset
        features: List of feature names
        notes: Optional notes
        
    Returns:
        Updated configuration
    """
    config['training']['last_trained'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    config['training']['training_duration_seconds'] = round(training_duration, 2)
    config['training']['dataset_size'] = dataset_size
    
    config['metrics']['accuracy'] = round(metrics['accuracy'], 4)
    config['metrics']['f1_score'] = round(metrics['f1_score'], 4)
    config['metrics']['precision'] = round(metrics['precision'], 4)
    config['metrics']['recall'] = round(metrics['recall'], 4)
    
    config['features'] = features
    
    if notes:
        config['notes'] = notes
    
    # Increment version
    version_parts = config['model']['version'].split('.')
    version_parts[-1] = str(int(version_parts[-1]) + 1)
    config['model']['version'] = '.'.join(version_parts)
    
    return config


def validate_dataset(df: pd.DataFrame, target_column: str) -> tuple:
    """
    Validate dataset structure.
    
    Args:
        df: Dataset DataFrame
        target_column: Name of target column
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "Dataset is empty"
    
    if target_column not in df.columns:
        return False, f"Target column '{target_column}' not found in dataset"
    
    if df[target_column].isna().all():
        return False, f"Target column '{target_column}' contains only null values"
    
    return True, ""
