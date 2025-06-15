"""
Utility functions for the bus demand forecasting project
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """Set up logging configuration"""
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging_config = {
        'level': getattr(logging, log_level.upper()),
        'format': log_format,
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }
    
    if log_file:
        logging_config['filename'] = log_file
        logging_config['filemode'] = 'w'
    
    logging.basicConfig(**logging_config)
    
    # Reduce noise from other libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('seaborn').setLevel(logging.WARNING)

def create_directories(project_root: Path) -> None:
    """Create project directory structure"""
    
    directories = [
        'data/raw',
        'data/processed', 
        'data/submissions',
        'models',
        'logs',
        'plots',
        'notebooks'
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Created project directories in {project_root}")

def load_and_validate_config(config_path: Path = None) -> Dict:
    """Load and validate configuration"""
    # This could be expanded to load from JSON/YAML config files
    from .config import MODEL_CONFIG, ENSEMBLE_WEIGHTS, FEATURE_CONFIG, CV_CONFIG
    
    config = {
        'model_config': MODEL_CONFIG,
        'ensemble_weights': ENSEMBLE_WEIGHTS,
        'feature_config': FEATURE_CONFIG,
        'cv_config': CV_CONFIG
    }
    
    # Validate configuration
    assert all(isinstance(w, (int, float)) for w in ENSEMBLE_WEIGHTS.values()), "Ensemble weights must be numeric"
    assert abs(sum(ENSEMBLE_WEIGHTS.values()) - 1.0) < 0.01, "Ensemble weights should sum to approximately 1"
    
    return config

def save_submission(predictions: np.ndarray, test_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Save predictions in submission format"""
    
    submission = pd.DataFrame({
        'route_key': test_df['route_key'],
        'final_seatcount': predictions
    })
    
    # Validate submission format
    assert len(submission) == len(test_df), "Submission length mismatch"
    assert not submission['route_key'].isnull().any(), "Missing route keys"
    assert not submission['final_seatcount'].isnull().any(), "Missing predictions"
    assert (submission['final_seatcount'] >= 0).all(), "Negative predictions found"
    
    submission.to_csv(output_path, index=False)
    
    logging.info(f"Submission saved to {output_path}")
    logging.info(f"Submission shape: {submission.shape}")
    logging.info(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    logging.info(f"Mean prediction: {predictions.mean():.2f}")
    
    return submission

def create_prediction_summary(predictions_dict: Dict[str, np.ndarray], ensemble_pred: np.ndarray) -> pd.DataFrame:
    """Create summary of all model predictions"""
    
    summary_df = pd.DataFrame(predictions_dict)
    summary_df['ensemble'] = ensemble_pred
    
    # Add statistical summaries
    summary_stats = summary_df.describe().T
    summary_stats['range'] = summary_stats['max'] - summary_stats['min']
    
    return summary_stats

def plot_prediction_distributions(predictions_dict: Dict[str, np.ndarray], output_dir: Path) -> None:
    """Plot distribution of predictions from different models"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subplot for each model
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(15, 10))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for i, (model_name, predictions) in enumerate(predictions_dict.items()):
        axes[i].hist(predictions, bins=50, alpha=0.7, label=model_name)
        axes[i].set_title(f'{model_name} Predictions')
        axes[i].set_xlabel('Predicted Seat Count')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    # Remove extra subplots
    for i in range(n_models, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    for model_name, predictions in predictions_dict.items():
        plt.hist(predictions, bins=50, alpha=0.5, label=model_name, density=True)
    
    plt.xlabel('Predicted Seat Count')
    plt.ylabel('Density')
    plt.title('Prediction Distributions Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'prediction_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Prediction distribution plots saved to {output_dir}")

def plot_feature_importance(feature_importance_dict: Dict[str, pd.DataFrame], output_dir: Path, top_n: int = 20) -> None:
    """Plot feature importance from different models"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot individual model feature importance
    for model_name, importance_df in feature_importance_dict.items():
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Features - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_dir / f'feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info(f"Feature importance plots saved to {output_dir}")

def plot_model_correlations(predictions_dict: Dict[str, np.ndarray], output_dir: Path) -> None:
    """Plot correlation matrix between model predictions"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pred_df = pd.DataFrame(predictions_dict)
    correlation_matrix = pred_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, cbar_kws={'label': 'Correlation'})
    plt.title('Model Prediction Correlations')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Model correlation plot saved to {output_dir}")

def analyze_data_quality(train_df: pd.DataFrame, test_df: pd.DataFrame, transactions_df: pd.DataFrame) -> Dict:
    """Analyze data quality and provide summary statistics"""
    
    analysis = {}
    
    # Missing values analysis
    analysis['missing_values'] = {
        'train': train_df.isnull().sum().to_dict(),
        'test': test_df.isnull().sum().to_dict(),
        'transactions': transactions_df.isnull().sum().to_dict()
    }
    
    # Data types
    analysis['data_types'] = {
        'train': train_df.dtypes.to_dict(),
        'test': test_df.dtypes.to_dict(),
        'transactions': transactions_df.dtypes.to_dict()
    }
    
    # Target variable analysis (train only)
    if 'final_seatcount' in train_df.columns:
        target = train_df['final_seatcount']
        analysis['target_analysis'] = {
            'mean': float(target.mean()),
            'median': float(target.median()),
            'std': float(target.std()),
            'min': float(target.min()),
            'max': float(target.max()),
            'zeros': int((target == 0).sum()),
            'negative': int((target < 0).sum()),
            'outliers_iqr': int(((target < target.quantile(0.25) - 1.5 * (target.quantile(0.75) - target.quantile(0.25))) |
                                (target > target.quantile(0.75) + 1.5 * (target.quantile(0.75) - target.quantile(0.25)))).sum())
        }
    
    # Route analysis
    train_routes = set(zip(train_df['srcid'], train_df['destid']))
    test_routes = set(zip(test_df['srcid'], test_df['destid']))
    
    analysis['route_analysis'] = {
        'unique_routes_train': len(train_routes),
        'unique_routes_test': len(test_routes),
        'common_routes': len(train_routes.intersection(test_routes)),
        'unseen_routes_in_test': len(test_routes - train_routes),
        'coverage_percentage': len(train_routes.intersection(test_routes)) / len(test_routes) * 100
    }
    
    # Date range analysis
    if 'doj' in train_df.columns:
        analysis['date_analysis'] = {
            'train_date_range': [str(train_df['doj'].min()), str(train_df['doj'].max())],
            'test_date_range': [str(test_df['doj'].min()), str(test_df['doj'].max())],
            'train_unique_dates': len(train_df['doj'].unique()),
            'test_unique_dates': len(test_df['doj'].unique())
        }
    
    return analysis

def calculate_baseline_metrics(train_df: pd.DataFrame) -> Dict:
    """Calculate baseline prediction metrics for comparison"""
    
    baselines = {}
    
    if 'final_seatcount' in train_df.columns:
        target = train_df['final_seatcount']
        
        # Global baselines
        baselines['global_mean'] = float(target.mean())
        baselines['global_median'] = float(target.median())
        
        # Route-based baselines
        route_means = train_df.groupby(['srcid', 'destid'])['final_seatcount'].mean()
        baselines['route_mean_baseline'] = float(route_means.mean())
        
        # Date-based baselines (if applicable)
        if 'doj' in train_df.columns:
            # Weekend vs weekday
            train_df_copy = train_df.copy()
            train_df_copy['is_weekend'] = train_df_copy['doj'].dt.dayofweek >= 5
            weekend_mean = train_df_copy[train_df_copy['is_weekend']]['final_seatcount'].mean()
            weekday_mean = train_df_copy[~train_df_copy['is_weekend']]['final_seatcount'].mean()
            
            baselines['weekend_mean'] = float(weekend_mean)
            baselines['weekday_mean'] = float(weekday_mean)
    
    return baselines

def validate_submission_format(submission_df: pd.DataFrame, test_df: pd.DataFrame) -> List[str]:
    """Validate submission format and return list of issues"""
    
    issues = []
    
    # Check required columns
    required_columns = ['route_key', 'final_seatcount']
    for col in required_columns:
        if col not in submission_df.columns:
            issues.append(f"Missing required column: {col}")
    
    if not issues:  # Only continue if required columns exist
        
        # Check length
        if len(submission_df) != len(test_df):
            issues.append(f"Submission length ({len(submission_df)}) doesn't match test length ({len(test_df)})")
        
        # Check for missing values
        if submission_df['route_key'].isnull().any():
            issues.append("Missing route_key values found")
        
        if submission_df['final_seatcount'].isnull().any():
            issues.append("Missing final_seatcount values found")
        
        # Check for negative predictions
        if (submission_df['final_seatcount'] < 0).any():
            negative_count = (submission_df['final_seatcount'] < 0).sum()
            issues.append(f"Found {negative_count} negative predictions")
        
        # Check for infinite values
        if np.isinf(submission_df['final_seatcount']).any():
            inf_count = np.isinf(submission_df['final_seatcount']).sum()
            issues.append(f"Found {inf_count} infinite predictions")
        
        # Check route key consistency
        if set(submission_df['route_key']) != set(test_df['route_key']):
            issues.append("Route keys don't match between submission and test set")
        
        # Check for extremely large predictions (potential outliers)
        if (submission_df['final_seatcount'] > 10000).any():
            large_count = (submission_df['final_seatcount'] > 10000).sum()
            issues.append(f"Found {large_count} predictions > 10000 (potential outliers)")
    
    return issues

def create_model_performance_report(cv_scores: Dict, output_path: Path) -> None:
    """Create a comprehensive model performance report"""
    
    report_lines = []
    report_lines.append("="*60)
    report_lines.append("MODEL PERFORMANCE REPORT")
    report_lines.append("="*60)
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Model performance summary
    report_lines.append("CROSS-VALIDATION RESULTS:")
    report_lines.append("-" * 30)
    
    for model_name, scores in cv_scores.items():
        mean_rmse = scores.get('mean_rmse', 'N/A')
        std_rmse = scores.get('std_rmse', 'N/A')
        
        if isinstance(mean_rmse, (int, float)) and isinstance(std_rmse, (int, float)):
            report_lines.append(f"{model_name:12} | RMSE: {mean_rmse:.4f} ± {std_rmse*2:.4f}")
        else:
            report_lines.append(f"{model_name:12} | RMSE: {mean_rmse}")
    
    report_lines.append("")
    
    # Best model identification
    if cv_scores:
        best_model = min(cv_scores.keys(), 
                        key=lambda x: cv_scores[x].get('mean_rmse', float('inf')))
        best_rmse = cv_scores[best_model].get('mean_rmse', 'N/A')
        report_lines.append(f"BEST SINGLE MODEL: {best_model} (RMSE: {best_rmse:.4f})")
        report_lines.append("")
    
    # Model comparison
    report_lines.append("MODEL RANKING (by CV RMSE):")
    report_lines.append("-" * 30)
    
    sorted_models = sorted(cv_scores.items(), 
                          key=lambda x: x[1].get('mean_rmse', float('inf')))
    
    for i, (model_name, scores) in enumerate(sorted_models, 1):
        mean_rmse = scores.get('mean_rmse', 'N/A')
        report_lines.append(f"{i}. {model_name} - {mean_rmse:.4f}")
    
    report_lines.append("")
    report_lines.append("="*60)
    
    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logging.info(f"Performance report saved to {output_path}")

def memory_usage_check(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Check and log memory usage of DataFrame"""
    
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    logging.info(f"{name} memory usage: {memory_mb:.2f} MB")
    
    # Log column-wise memory usage for large DataFrames
    if memory_mb > 100:  # If larger than 100MB
        col_memory = df.memory_usage(deep=True) / 1024**2
        top_memory_cols = col_memory.nlargest(5)
        logging.info(f"Top 5 memory-consuming columns in {name}:")
        for col, mem in top_memory_cols.items():
            logging.info(f"  {col}: {mem:.2f} MB")

def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by downcasting numeric types"""
    
    df_optimized = df.copy()
    
    # Optimize integer columns
    for col in df_optimized.select_dtypes(include=['int64']).columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
    
    # Optimize float columns
    for col in df_optimized.select_dtypes(include=['float64']).columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
    
    # Convert object columns to category if beneficial
    for col in df_optimized.select_dtypes(include=['object']).columns:
        unique_ratio = len(df_optimized[col].unique()) / len(df_optimized)
        if unique_ratio < 0.5:  # If less than 50% unique values
            df_optimized[col] = df_optimized[col].astype('category')
    
    # Log memory savings
    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2
    savings = (1 - optimized_memory / original_memory) * 100
    
    logging.info(f"Memory optimization: {original_memory:.2f}MB -> {optimized_memory:.2f}MB ({savings:.1f}% reduction)")
    
    return df_optimized