import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from src.constants import ALL_MODEL_FEATURES, BINARY_NUMERIC_FEATURES, NUMERIC_FEATURES, STRING_CATEGORICALS
from sklearn.model_selection import train_test_split


preprocessor = ColumnTransformer([
    ('num', StandardScaler(), NUMERIC_FEATURES),
    ('cat', OrdinalEncoder(), STRING_CATEGORICALS + BINARY_NUMERIC_FEATURES)
])

def apply_preprocessing(X_train, X_test):
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test