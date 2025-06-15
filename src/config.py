"""
Configuration file for Bus Demand Forecasting
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SUBMISSIONS_DIR = DATA_DIR / "submissions"

# Data files
TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test_8gqdJqH.csv"
TRANSACTIONS_FILE = RAW_DATA_DIR / "transactions.csv"

# Model parameters
MODEL_CONFIG = {
    'lgb': {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 8,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    },
    'xgb': {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    },
    'rf': {
        'n_estimators': 500,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    },
    'gb': {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 8,
        'random_state': 42
    },
    'ridge': {
        'alpha': 10.0,
        'random_state': 42
    }
}

# Ensemble weights
ENSEMBLE_WEIGHTS = {
    'lgb': 0.3,
    'xgb': 0.25,
    'rf': 0.2,
    'gb': 0.15,
    'ridge': 0.1
}

# Feature engineering parameters
FEATURE_CONFIG = {
    'lag_periods': [1, 2, 3, 7, 14],
    'rolling_windows': [3, 7, 14],
    'dbd_filter': 15,  # Days before departure for prediction
    'holiday_proximity_days': 3,
    'date_format': '%Y-%m-%d'  # Fixed: Your data uses YYYY-MM-DD format
}

# Cross-validation parameters
CV_CONFIG = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': 42
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}