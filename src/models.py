"""
Model definitions and training module
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import joblib
from pathlib import Path

from .config import MODEL_CONFIG, CV_CONFIG

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class for training and managing multiple models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        self.cv_scores = {}
        self.model_config = MODEL_CONFIG
        self.cv_config = CV_CONFIG
        
    def prepare_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Prepare features for modeling"""
        logger.info("Preparing features for modeling")
        
        # Select features (exclude target and identifier columns)
        exclude_cols = ['final_seatcount', 'doj', 'route_key', 'doi']
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        logger.info(f"Selected {len(feature_cols)} features for modeling")
        
        # Handle categorical variables
        categorical_cols = train_df[feature_cols].select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            # Fit on combined data to handle unseen categories
            combined_data = pd.concat([train_df[col], test_df[col]], ignore_index=True)
            le.fit(combined_data.astype(str))
            
            train_df[col] = le.transform(train_df[col].astype(str))
            test_df[col] = le.transform(test_df[col].astype(str))
            self.label_encoders[col] = le
        
        X_train = train_df[feature_cols]
        y_train = train_df['final_seatcount']
        X_test = test_df[feature_cols]
        
        # Feature scaling for linear models
        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Store scaled versions
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        
        return X_train, y_train, X_test
    
    def create_models(self) -> Dict:
        """Create all models with specified configurations"""
        logger.info("Creating models")
        
        # LightGBM
        self.models['lgb'] = lgb.LGBMRegressor(**self.model_config['lgb'])
        
        # XGBoost
        self.models['xgb'] = xgb.XGBRegressor(**self.model_config['xgb'])
        
        # Random Forest
        self.models['rf'] = RandomForestRegressor(**self.model_config['rf'])
        
        # Gradient Boosting
        self.models['gb'] = GradientBoostingRegressor(**self.model_config['gb'])
        
        # Ridge Regression
        self.models['ridge'] = Ridge(**self.model_config['ridge'])
        
        # Lasso Regression
        self.models['lasso'] = Lasso(alpha=1.0, random_state=42)
        
        logger.info(f"Created {len(self.models)} models: {list(self.models.keys())}")
        
        return self.models
    
    def train_single_model(self, name: str, model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train a single model with cross-validation"""
        logger.info(f"Training {name} model")
        
        # Use scaled features for linear models
        if name in ['ridge', 'lasso']:
            X_use = self.X_train_scaled
        else:
            X_use = X_train
        
        # Cross-validation
        kf = KFold(**self.cv_config)
        cv_scores = cross_val_score(
            model, X_use, y_train, 
            cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        
        cv_rmse = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        logger.info(f"{name} CV RMSE: {cv_rmse:.4f} (+/- {cv_std * 2:.4f})")
        
        # Train on full dataset
        model.fit(X_use, y_train)
        
        # Feature importance for tree-based models
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_use.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.feature_importance[name] = feature_importance
            logger.info(f"Top 5 features for {name}:")
            logger.info(feature_importance.head().to_string())
        
        # Store CV scores
        self.cv_scores[name] = {
            'mean_rmse': cv_rmse,
            'std_rmse': cv_std,
            'scores': cv_scores
        }
        
        return {
            'model': model,
            'cv_rmse': cv_rmse,
            'cv_std': cv_std,
            'feature_importance': feature_importance
        }
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train all models"""
        logger.info("Training all models")
        
        results = {}
        
        for name, model in self.models.items():
            try:
                result = self.train_single_model(name, model, X_train, y_train)
                results[name] = result
                logger.info(f"Successfully trained {name}")
            except Exception as e:
                logger.error(f"Failed to train {name}: {str(e)}")
                continue
        
        # Log overall results
        logger.info("\n" + "="*50)
        logger.info("TRAINING RESULTS SUMMARY")
        logger.info("="*50)
        
        for name, result in results.items():
            logger.info(f"{name:8} - CV RMSE: {result['cv_rmse']:.4f} (+/- {result['cv_std']*2:.4f})")
        
        return results
    
    def predict_single_model(self, name: str, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions with a single model"""
        model = self.models[name]
        
        # Use scaled features for linear models
        if name in ['ridge', 'lasso']:
            X_use = self.X_test_scaled
        else:
            X_use = X_test
        
        predictions = model.predict(X_use)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def predict_all_models(self, X_test: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions with all trained models"""
        logger.info("Making predictions with all models")
        
        predictions = {}
        
        for name in self.models.keys():
            try:
                pred = self.predict_single_model(name, X_test)
                predictions[name] = pred
                logger.info(f"{name} predictions - Range: [{pred.min():.2f}, {pred.max():.2f}], Mean: {pred.mean():.2f}")
            except Exception as e:
                logger.error(f"Failed to predict with {name}: {str(e)}")
                continue
        
        return predictions
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """Get aggregated feature importance across all models"""
        if not self.feature_importance:
            return pd.DataFrame()
        
        # Combine feature importance from all models
        all_importance = []
        
        for model_name, importance_df in self.feature_importance.items():
            importance_df_copy = importance_df.copy()
            importance_df_copy['model'] = model_name
            all_importance.append(importance_df_copy)
        
        combined_importance = pd.concat(all_importance, ignore_index=True)
        
        # Calculate mean importance across models
        mean_importance = combined_importance.groupby('feature')['importance'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        mean_importance = mean_importance.sort_values('mean', ascending=False)
        
        return mean_importance
    
    def save_models(self, output_dir: Path) -> None:
        """Save all trained models and preprocessors"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            model_path = output_dir / f"{name}_model.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} model to {model_path}")
        
        # Save preprocessors
        scaler_path = output_dir / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        
        encoders_path = output_dir / "label_encoders.joblib"
        joblib.dump(self.label_encoders, encoders_path)
        
        # Save feature importance
        if self.feature_importance:
            importance_summary = self.get_feature_importance_summary()
            importance_path = output_dir / "feature_importance.csv"
            importance_summary.to_csv(importance_path, index=False)
        
        # Save CV scores
        cv_scores_path = output_dir / "cv_scores.joblib"
        joblib.dump(self.cv_scores, cv_scores_path)
        
        logger.info(f"All models and preprocessors saved to {output_dir}")
    
    def load_models(self, input_dir: Path) -> None:
        """Load pre-trained models and preprocessors"""
        logger.info(f"Loading models from {input_dir}")
        
        # Load models
        for name in ['lgb', 'xgb', 'rf', 'gb', 'ridge', 'lasso']:
            model_path = input_dir / f"{name}_model.joblib"
            if model_path.exists():
                self.models[name] = joblib.load(model_path)
                logger.info(f"Loaded {name} model")
        
        # Load preprocessors
        scaler_path = input_dir / "scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        encoders_path = input_dir / "label_encoders.joblib"
        if encoders_path.exists():
            self.label_encoders = joblib.load(encoders_path)
        
        # Load CV scores
        cv_scores_path = input_dir / "cv_scores.joblib"
        if cv_scores_path.exists():
            self.cv_scores = joblib.load(cv_scores_path)
        
        logger.info("Model loading completed")

class ModelValidator:
    """Class for advanced model validation and analysis"""
    
    def __init__(self, trainer: ModelTrainer):
        self.trainer = trainer
    
    def validate_predictions(self, predictions: Dict[str, np.ndarray], X_test: pd.DataFrame) -> Dict:
        """Validate model predictions and check for issues"""
        validation_results = {}
        
        for name, pred in predictions.items():
            results = {
                'negative_predictions': (pred < 0).sum(),
                'zero_predictions': (pred == 0).sum(),
                'extreme_predictions': (pred > 1000).sum(),  # Assuming 1000 is unusually high
                'nan_predictions': np.isnan(pred).sum(),
                'prediction_range': [pred.min(), pred.max()],
                'prediction_stats': {
                    'mean': pred.mean(),
                    'median': np.median(pred),
                    'std': pred.std()
                }
            }
            
            validation_results[name] = results
        
        return validation_results
    
    def analyze_prediction_consistency(self, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Analyze consistency between different model predictions"""
        pred_df = pd.DataFrame(predictions)
        
        # Calculate correlation between models
        correlation_matrix = pred_df.corr()
        
        # Calculate prediction statistics
        pred_df['mean'] = pred_df.mean(axis=1)
        pred_df['std'] = pred_df.std(axis=1)
        pred_df['cv'] = pred_df['std'] / pred_df['mean']  # Coefficient of variation
        
        consistency_stats = {
            'mean_correlation': correlation_matrix.mean().mean(),
            'min_correlation': correlation_matrix.min().min(),
            'high_variance_predictions': (pred_df['cv'] > 0.5).sum(),
            'prediction_agreement': correlation_matrix
        }
        
        return consistency_stats