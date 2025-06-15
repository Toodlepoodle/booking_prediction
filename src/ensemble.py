"""
Ensemble methods and final prediction generation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from .config import ENSEMBLE_WEIGHTS

logger = logging.getLogger(__name__)

class EnsemblePredictor:
    """Class for creating ensemble predictions"""
    
    def __init__(self, base_weights: Dict[str, float] = None):
        self.base_weights = base_weights or ENSEMBLE_WEIGHTS
        self.optimized_weights = None
        self.ensemble_methods = ['simple_average', 'weighted_average', 'rank_average', 'optimized_weights']
        
    def simple_average_ensemble(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple average of all model predictions"""
        pred_matrix = np.column_stack(list(predictions.values()))
        return np.mean(pred_matrix, axis=1)
    
    def weighted_average_ensemble(self, predictions: Dict[str, np.ndarray], weights: Dict[str, float] = None) -> np.ndarray:
        """Weighted average ensemble using specified weights"""
        if weights is None:
            weights = self.base_weights
        
        ensemble_pred = np.zeros(len(list(predictions.values())[0]))
        total_weight = 0
        
        for model_name, pred in predictions.items():
            if model_name in weights:
                weight = weights[model_name]
                ensemble_pred += weight * pred
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return ensemble_pred
    
    def rank_average_ensemble(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Rank-based ensemble (average of rankings)"""
        pred_df = pd.DataFrame(predictions)
        
        # Convert predictions to ranks
        rank_df = pred_df.rank(method='average')
        
        # Average ranks and convert back to predictions
        avg_ranks = rank_df.mean(axis=1)
        
        # Convert ranks back to approximate predictions
        # Use the overall distribution of predictions
        all_preds = np.concatenate(list(predictions.values()))
        sorted_preds = np.sort(all_preds)
        
        # Map average ranks to prediction values
        ensemble_pred = np.interp(
            avg_ranks, 
            np.arange(1, len(avg_ranks) + 1), 
            np.sort(pred_df.mean(axis=1))
        )
        
        return ensemble_pred
    
    def optimize_weights(
        self, 
        predictions: Dict[str, np.ndarray], 
        y_true: np.ndarray, 
        method: str = 'RMSE'
    ) -> Dict[str, float]:
        """Optimize ensemble weights using cross-validation or validation set"""
        logger.info("Optimizing ensemble weights")
        
        model_names = list(predictions.keys())
        pred_matrix = np.column_stack([predictions[name] for name in model_names])
        
        def objective(weights):
            weights = np.abs(weights)  # Ensure positive weights
            weights /= np.sum(weights)  # Normalize
            
            ensemble_pred = np.dot(pred_matrix, weights)
            
            if method == 'RMSE':
                return np.sqrt(mean_squared_error(y_true, ensemble_pred))
            elif method == 'MAE':
                return np.mean(np.abs(y_true - ensemble_pred))
            else:
                raise ValueError(f"Unknown optimization method: {method}")
        
        # Initial weights (equal weighting)
        initial_weights = np.ones(len(model_names)) / len(model_names)
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1}
        bounds = [(0, 1) for _ in range(len(model_names))]
        
        # Optimize
        result = minimize(
            objective, 
            initial_weights, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimized_weights = dict(zip(model_names, result.x))
            self.optimized_weights = optimized_weights
            
            logger.info("Weight optimization successful!")
            logger.info("Optimized weights:")
            for name, weight in optimized_weights.items():
                logger.info(f"  {name}: {weight:.4f}")
            
            return optimized_weights
        else:
            logger.warning("Weight optimization failed, using default weights")
            return self.base_weights
    
    def optimized_weights_ensemble(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Ensemble using optimized weights"""
        if self.optimized_weights is None:
            logger.warning("No optimized weights available, using base weights")
            return self.weighted_average_ensemble(predictions, self.base_weights)
        
        return self.weighted_average_ensemble(predictions, self.optimized_weights)
    
    def create_all_ensembles(self, predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Create all types of ensemble predictions"""
        logger.info("Creating ensemble predictions")
        
        ensembles = {}
        
        # Simple average
        ensembles['simple_average'] = self.simple_average_ensemble(predictions)
        
        # Weighted average
        ensembles['weighted_average'] = self.weighted_average_ensemble(predictions)
        
        # Rank average
        ensembles['rank_average'] = self.rank_average_ensemble(predictions)
        
        # Optimized weights (if available)
        if self.optimized_weights is not None:
            ensembles['optimized_weights'] = self.optimized_weights_ensemble(predictions)
        
        # Log ensemble statistics
        for name, pred in ensembles.items():
            logger.info(f"{name} - Range: [{pred.min():.2f}, {pred.max():.2f}], Mean: {pred.mean():.2f}")
        
        return ensembles
    
    def select_best_ensemble(
        self, 
        ensembles: Dict[str, np.ndarray], 
        predictions: Dict[str, np.ndarray],
        selection_method: str = 'cv_score'
    ) -> Tuple[str, np.ndarray]:
        """Select the best ensemble method"""
        
        if selection_method == 'cv_score' and hasattr(self, 'cv_scores'):
            # Use CV scores to weight ensemble selection
            best_method = 'weighted_average'  # Default fallback
            
        elif selection_method == 'prediction_stability':
            # Select based on prediction stability (lowest variance across models)
            pred_df = pd.DataFrame(predictions)
            stability_scores = {}
            
            for ens_name, ens_pred in ensembles.items():
                # Calculate how well ensemble represents individual predictions
                pred_df['ensemble'] = ens_pred
                correlations = pred_df.corr()['ensemble'].drop('ensemble')
                stability_scores[ens_name] = correlations.mean()
            
            best_method = max(stability_scores, key=stability_scores.get)
            logger.info(f"Selected {best_method} based on stability (score: {stability_scores[best_method]:.4f})")
            
        else:
            # Default to weighted average
            best_method = 'weighted_average'
        
        return best_method, ensembles[best_method]

class AdvancedEnsemble:
    """Advanced ensemble methods for improved predictions"""
    
    def __init__(self):
        self.meta_model = None
        self.stacking_features = None
    
    def create_stacking_features(self, predictions: Dict[str, np.ndarray], X_test: pd.DataFrame) -> pd.DataFrame:
        """Create features for stacking ensemble"""
        
        # Base predictions as features
        stacking_df = pd.DataFrame(predictions)
        
        # Add statistical features
        stacking_df['pred_mean'] = stacking_df.mean(axis=1)
        stacking_df['pred_std'] = stacking_df.std(axis=1)
        stacking_df['pred_min'] = stacking_df.min(axis=1)
        stacking_df['pred_max'] = stacking_df.max(axis=1)
        stacking_df['pred_range'] = stacking_df['pred_max'] - stacking_df['pred_min']
        
        # Add model agreement features
        pred_array = stacking_df[list(predictions.keys())].values
        stacking_df['agreement_score'] = np.std(pred_array, axis=1) / (np.mean(pred_array, axis=1) + 1e-8)
        
        # Add selected original features that might help meta-learning
        important_features = ['seats_15d_before', 'searches_15d_before', 'doj_is_weekend', 'doj_is_holiday']
        for feature in important_features:
            if feature in X_test.columns:
                stacking_df[feature] = X_test[feature].values
        
        return stacking_df
    
    def bayesian_model_averaging(self, predictions: Dict[str, np.ndarray], cv_scores: Dict = None) -> np.ndarray:
        """Bayesian Model Averaging based on model performance"""
        
        if cv_scores is None:
            # If no CV scores, use equal weights
            return np.mean(np.column_stack(list(predictions.values())), axis=1)
        
        # Convert RMSE to weights (lower RMSE = higher weight)
        rmse_scores = {name: scores.get('mean_rmse', 1.0) for name, scores in cv_scores.items()}
        
        # Convert to weights (inverse of RMSE)
        weights = {}
        for name in predictions.keys():
            if name in rmse_scores:
                weights[name] = 1.0 / rmse_scores[name]
            else:
                weights[name] = 1.0
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {name: w / total_weight for name, w in weights.items()}
        
        # Create ensemble
        ensemble_pred = np.zeros(len(list(predictions.values())[0]))
        for name, pred in predictions.items():
            if name in weights:
                ensemble_pred += weights[name] * pred
        
        logger.info("BMA weights:")
        for name, weight in weights.items():
            logger.info(f"  {name}: {weight:.4f}")
        
        return ensemble_pred
    
    def dynamic_ensemble(self, predictions: Dict[str, np.ndarray], X_test: pd.DataFrame) -> np.ndarray:
        """Dynamic ensemble that adapts weights based on input features"""
        
        # Simple dynamic weighting based on route characteristics
        dynamic_weights = np.ones((len(X_test), len(predictions)))
        
        # Adjust weights based on feature values
        for i, model_name in enumerate(predictions.keys()):
            if model_name == 'lgb':
                # LightGBM might be better for complex patterns
                if 'same_region' in X_test.columns:
                    complex_routes = X_test['same_region'] == 0
                    dynamic_weights[complex_routes, i] *= 1.2
            
            elif model_name == 'rf':
                # Random Forest might be better for weekend/holiday patterns
                if 'doj_is_weekend' in X_test.columns:
                    weekend_routes = X_test['doj_is_weekend'] == 1
                    dynamic_weights[weekend_routes, i] *= 1.1
        
        # Normalize weights for each prediction
        dynamic_weights = dynamic_weights / dynamic_weights.sum(axis=1, keepdims=True)
        
        # Create ensemble
        pred_matrix = np.column_stack(list(predictions.values()))
        ensemble_pred = np.sum(pred_matrix * dynamic_weights, axis=1)
        
        return ensemble_pred

def create_final_prediction(
    predictions: Dict[str, np.ndarray], 
    X_test: pd.DataFrame = None,
    cv_scores: Dict = None,
    method: str = 'weighted_average'
) -> np.ndarray:
    """Create final ensemble prediction using specified method"""
    
    ensemble_predictor = EnsemblePredictor()
    
    if method == 'weighted_average':
        final_pred = ensemble_predictor.weighted_average_ensemble(predictions)
    elif method == 'simple_average':
        final_pred = ensemble_predictor.simple_average_ensemble(predictions)
    elif method == 'rank_average':
        final_pred = ensemble_predictor.rank_average_ensemble(predictions)
    elif method == 'bma':
        advanced_ensemble = AdvancedEnsemble()
        final_pred = advanced_ensemble.bayesian_model_averaging(predictions, cv_scores)
    elif method == 'dynamic':
        if X_test is None:
            logger.warning("X_test required for dynamic ensemble, falling back to weighted average")
            final_pred = ensemble_predictor.weighted_average_ensemble(predictions)
        else:
            advanced_ensemble = AdvancedEnsemble()
            final_pred = advanced_ensemble.dynamic_ensemble(predictions, X_test)
    else:
        logger.warning(f"Unknown ensemble method {method}, using weighted average")
        final_pred = ensemble_predictor.weighted_average_ensemble(predictions)
    
    # Ensure non-negative predictions
    final_pred = np.maximum(final_pred, 0)
    
    logger.info(f"Final prediction stats - Range: [{final_pred.min():.2f}, {final_pred.max():.2f}], Mean: {final_pred.mean():.2f}")
    
    return final_pred