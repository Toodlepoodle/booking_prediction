"""
Bus Demand Forecasting Package

A comprehensive machine learning solution for predicting bus journey demand.
"""

__version__ = "1.0.0"
__author__ = "Bus Demand Forecasting Team"
__email__ = "team@busforecasting.com"

# Import main classes for easy access
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .models import ModelTrainer, ModelValidator
from .ensemble import EnsemblePredictor, create_final_prediction
from .utils import setup_logging, save_submission

__all__ = [
    'DataLoader',
    'FeatureEngineer', 
    'ModelTrainer',
    'ModelValidator',
    'EnsemblePredictor',
    'create_final_prediction',
    'setup_logging',
    'save_submission'
]