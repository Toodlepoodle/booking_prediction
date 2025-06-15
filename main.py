"""
Main execution script for Bus Demand Forecasting
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import sys
import argparse

# Import project modules
from src.config import PROJECT_ROOT, SUBMISSIONS_DIR, PROCESSED_DATA_DIR
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.models import ModelTrainer, ModelValidator
from src.ensemble import create_final_prediction
from src.utils import (
    setup_logging, create_directories, save_submission, 
    create_prediction_summary, plot_prediction_distributions,
    plot_feature_importance, plot_model_correlations,
    analyze_data_quality, validate_submission_format,
    create_model_performance_report, memory_usage_check
)

class BusDemandForecaster:
    """Main forecasting pipeline class"""
    
    def __init__(self, ensemble_method='weighted_average', save_models=True, create_plots=True):
        self.ensemble_method = ensemble_method
        self.save_models = save_models
        self.create_plots = create_plots
        
        # Initialize components
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.model_validator = ModelValidator(self.model_trainer)
        
        # Data containers
        self.train_df = None
        self.test_df = None
        self.transactions_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        
        # Results
        self.predictions = {}
        self.ensemble_prediction = None
        self.submission = None
        
    def setup_environment(self):
        """Set up project environment"""
        logging.info("Setting up project environment")
        
        # Create directories
        create_directories(PROJECT_ROOT)
        
        # Setup logging with file output
        log_dir = PROJECT_ROOT / "logs"
        log_file = log_dir / f"bus_forecasting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        setup_logging('INFO', str(log_file))
        
        logging.info("="*60)
        logging.info("BUS DEMAND FORECASTING PIPELINE STARTED")
        logging.info("="*60)
        logging.info(f"Project root: {PROJECT_ROOT}")
        logging.info(f"Ensemble method: {self.ensemble_method}")
        logging.info(f"Save models: {self.save_models}")
        logging.info(f"Create plots: {self.create_plots}")
        
    def load_and_validate_data(self):
        """Load and validate all datasets"""
        logging.info("STEP 1: Loading and validating data")
        
        # Load data
        self.train_df, self.test_df, self.transactions_df = self.data_loader.load_all_data()
        
        # Memory usage check
        memory_usage_check(self.train_df, "Train DataFrame")
        memory_usage_check(self.test_df, "Test DataFrame") 
        memory_usage_check(self.transactions_df, "Transactions DataFrame")
        
        # Data quality analysis
        data_analysis = analyze_data_quality(self.train_df, self.test_df, self.transactions_df)
        
        # Log key insights
        logging.info("Data Quality Summary:")
        logging.info(f"  Route coverage: {data_analysis['route_analysis']['coverage_percentage']:.1f}%")
        logging.info(f"  Unseen routes in test: {data_analysis['route_analysis']['unseen_routes_in_test']}")
        
        if 'target_analysis' in data_analysis:
            target_stats = data_analysis['target_analysis']
            logging.info(f"  Target mean: {target_stats['mean']:.2f}")
            logging.info(f"  Target zeros: {target_stats['zeros']}")
            logging.info(f"  Target outliers: {target_stats['outliers_iqr']}")
        
        # Save data summary
        summary_path = PROCESSED_DATA_DIR / "data_analysis_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(str(data_analysis))
        
        logging.info("Data loading and validation completed!")
        
    def engineer_features(self):
        """Perform feature engineering"""
        logging.info("STEP 2: Feature engineering")
        
        # Create all features
        self.train_df, self.test_df = self.feature_engineer.create_all_features(
            self.train_df, self.test_df, self.transactions_df
        )
        
        # Memory optimization after feature engineering
        logging.info("Optimizing memory usage after feature engineering")
        memory_usage_check(self.train_df, "Train DataFrame (after features)")
        memory_usage_check(self.test_df, "Test DataFrame (after features)")
        
        # Save processed data
        if self.save_models:
            self.data_loader.train = self.train_df
            self.data_loader.test = self.test_df
            self.data_loader.save_processed_data(PROCESSED_DATA_DIR)
        
        logging.info("Feature engineering completed!")
        
    def prepare_modeling_data(self):
        """Prepare data for modeling"""
        logging.info("STEP 3: Preparing data for modeling")
        
        # Prepare features
        self.X_train, self.y_train, self.X_test = self.model_trainer.prepare_features(
            self.train_df, self.test_df
        )
        
        logging.info(f"Features prepared: {self.X_train.shape[1]} features")
        logging.info(f"Training samples: {len(self.X_train)}")
        logging.info(f"Test samples: {len(self.X_test)}")
        
    def train_models(self):
        """Train all models"""
        logging.info("STEP 4: Training models")
        
        # Create models
        self.model_trainer.create_models()
        
        # Train all models
        training_results = self.model_trainer.train_all_models(self.X_train, self.y_train)
        
        # Save models if requested
        if self.save_models:
            models_dir = PROJECT_ROOT / "models" / f"models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.model_trainer.save_models(models_dir)
            
            # Create performance report
            report_path = models_dir / "performance_report.txt"
            create_model_performance_report(self.model_trainer.cv_scores, report_path)
        
        logging.info("Model training completed!")
        
        return training_results
        
    def make_predictions(self):
        """Make predictions with all models"""
        logging.info("STEP 5: Making predictions")
        
        # Get individual model predictions
        self.predictions = self.model_trainer.predict_all_models(self.X_test)
        
        # Validate predictions
        validation_results = self.model_validator.validate_predictions(self.predictions, self.X_test)
        
        # Log validation results
        for model, results in validation_results.items():
            if results['negative_predictions'] > 0:
                logging.warning(f"{model}: {results['negative_predictions']} negative predictions")
            if results['extreme_predictions'] > 0:
                logging.warning(f"{model}: {results['extreme_predictions']} extreme predictions")
        
        # Create ensemble prediction
        self.ensemble_prediction = create_final_prediction(
            self.predictions, 
            self.X_test,
            self.model_trainer.cv_scores,
            method=self.ensemble_method
        )
        
        logging.info("Predictions completed!")
        
    def create_submission(self):
        """Create final submission"""
        logging.info("STEP 6: Creating submission")
        
        # Create submission file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        submission_path = SUBMISSIONS_DIR / f"submission_{self.ensemble_method}_{timestamp}.csv"
        
        self.submission = save_submission(
            self.ensemble_prediction, 
            self.test_df, 
            submission_path
        )
        
        # Validate submission format
        validation_issues = validate_submission_format(self.submission, self.test_df)
        
        if validation_issues:
            logging.warning("Submission validation issues found:")
            for issue in validation_issues:
                logging.warning(f"  - {issue}")
        else:
            logging.info("Submission validation passed!")
        
        # Create prediction summary
        summary_stats = create_prediction_summary(self.predictions, self.ensemble_prediction)
        summary_path = SUBMISSIONS_DIR / f"prediction_summary_{timestamp}.csv"
        summary_stats.to_csv(summary_path)
        
        logging.info(f"Submission created: {submission_path}")
        
        return submission_path
        
    def create_visualizations(self):
        """Create visualization plots"""
        if not self.create_plots:
            return
            
        logging.info("STEP 7: Creating visualizations")
        
        plots_dir = PROJECT_ROOT / "plots" / f"plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Prediction distributions
            plot_prediction_distributions(self.predictions, plots_dir)
            
            # Feature importance
            if self.model_trainer.feature_importance:
                plot_feature_importance(self.model_trainer.feature_importance, plots_dir)
            
            # Model correlations
            plot_model_correlations(self.predictions, plots_dir)
            
            logging.info(f"Visualizations saved to {plots_dir}")
            
        except Exception as e:
            logging.warning(f"Error creating visualizations: {str(e)}")
    
    def run_complete_pipeline(self):
        """Run the complete forecasting pipeline"""
        
        try:
            # Setup
            self.setup_environment()
            
            # Execute pipeline steps
            self.load_and_validate_data()
            self.engineer_features()
            self.prepare_modeling_data()
            self.train_models()
            self.make_predictions()
            submission_path = self.create_submission()
            self.create_visualizations()
            
            # Final summary
            logging.info("="*60)
            logging.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logging.info("="*60)
            logging.info(f"Final submission: {submission_path}")
            logging.info(f"Ensemble method: {self.ensemble_method}")
            logging.info(f"Final prediction stats:")
            logging.info(f"  Mean: {self.ensemble_prediction.mean():.2f}")
            logging.info(f"  Std: {self.ensemble_prediction.std():.2f}")
            logging.info(f"  Range: [{self.ensemble_prediction.min():.2f}, {self.ensemble_prediction.max():.2f}]")
            
            return submission_path
            
        except Exception as e:
            logging.error(f"Pipeline failed with error: {str(e)}")
            logging.error("Check logs for detailed error information")
            raise

def main():
    """Main function with command line argument support"""
    
    parser = argparse.ArgumentParser(description='Bus Demand Forecasting Pipeline')
    parser.add_argument('--ensemble-method', 
                       choices=['simple_average', 'weighted_average', 'rank_average', 'bma', 'dynamic'],
                       default='weighted_average',
                       help='Ensemble method to use')
    parser.add_argument('--no-save-models', action='store_true',
                       help='Skip saving trained models')
    parser.add_argument('--no-plots', action='store_true', 
                       help='Skip creating plots')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Initialize forecaster
    forecaster = BusDemandForecaster(
        ensemble_method=args.ensemble_method,
        save_models=not args.no_save_models,
        create_plots=not args.no_plots
    )
    
    # Run pipeline
    try:
        submission_path = forecaster.run_complete_pipeline()
        print(f"\n🎉 SUCCESS! Submission created: {submission_path}")
        print(f"📊 Check logs and plots in: {PROJECT_ROOT}")
        return 0
        
    except Exception as e:
        print(f"\n❌ FAILED: {str(e)}")
        print(f"📋 Check logs for details: {PROJECT_ROOT}/logs/")
        return 1

if __name__ == "__main__":
    exit_code = main()