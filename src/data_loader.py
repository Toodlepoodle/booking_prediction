"""
Data loading and basic preprocessing module
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict
from .config import TRAIN_FILE, TEST_FILE, TRANSACTIONS_FILE, FEATURE_CONFIG

logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading and basic preprocessing of data"""
    
    def __init__(self):
        self.train = None
        self.test = None
        self.transactions = None
        self.date_format = FEATURE_CONFIG['date_format']
        
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all datasets and perform basic preprocessing"""
        logger.info("Loading datasets...")
        
        # Load datasets
        self.train = pd.read_csv(TRAIN_FILE)
        self.test = pd.read_csv(TEST_FILE)
        self.transactions = pd.read_csv(TRANSACTIONS_FILE)
        
        logger.info(f"Train shape: {self.train.shape}")
        logger.info(f"Test shape: {self.test.shape}")
        logger.info(f"Transactions shape: {self.transactions.shape}")
        
        # Basic preprocessing
        self.train = self._preprocess_dates(self.train)
        self.test = self._preprocess_dates(self.test)
        self.transactions = self._preprocess_dates(self.transactions)
        
        # Data validation
        self._validate_data()
        
        return self.train, self.test, self.transactions
    
    def _preprocess_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert date columns to datetime"""
        df = df.copy()
        
        if 'doj' in df.columns:
            # Handle multiple date formats
            try:
                df['doj'] = pd.to_datetime(df['doj'], format=self.date_format)
            except ValueError:
                # If format doesn't match, try inferring
                df['doj'] = pd.to_datetime(df['doj'], infer_datetime_format=True)
                
        if 'doi' in df.columns:
            try:
                df['doi'] = pd.to_datetime(df['doi'], format=self.date_format)
            except ValueError:
                # If format doesn't match, try inferring
                df['doi'] = pd.to_datetime(df['doi'], infer_datetime_format=True)
            
        return df
    
    def _validate_data(self) -> None:
        """Validate loaded data"""
        logger.info("Validating data...")
        
        # Check for missing values in key columns
        key_columns = ['srcid', 'destid', 'doj']
        for col in key_columns:
            if col in self.train.columns:
                missing_count = self.train[col].isnull().sum()
                if missing_count > 0:
                    logger.warning(f"Missing values in train.{col}: {missing_count}")
            
            if col in self.test.columns:
                missing_count = self.test[col].isnull().sum()
                if missing_count > 0:
                    logger.warning(f"Missing values in test.{col}: {missing_count}")
        
        # Check date ranges
        if self.train is not None and 'doj' in self.train.columns:
            logger.info(f"Train date range: {self.train['doj'].min()} to {self.train['doj'].max()}")
        
        if self.test is not None and 'doj' in self.test.columns:
            logger.info(f"Test date range: {self.test['doj'].min()} to {self.test['doj'].max()}")
        
        # Check route consistency
        train_routes = set(zip(self.train['srcid'], self.train['destid']))
        test_routes = set(zip(self.test['srcid'], self.test['destid']))
        
        unseen_routes = test_routes - train_routes
        if unseen_routes:
            logger.warning(f"Found {len(unseen_routes)} unseen routes in test set")
        
        logger.info("Data validation completed!")
    
    def get_filtered_transactions(self, dbd_filter: int = None) -> pd.DataFrame:
        """Get transactions filtered by days before departure"""
        if dbd_filter is None:
            dbd_filter = FEATURE_CONFIG['dbd_filter']
        
        filtered_transactions = self.transactions[
            self.transactions['dbd'] == dbd_filter
        ].copy()
        
        logger.info(f"Filtered transactions for dbd={dbd_filter}: {len(filtered_transactions)} records")
        
        return filtered_transactions
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of loaded data"""
        summary = {
            'train_shape': self.train.shape if self.train is not None else None,
            'test_shape': self.test.shape if self.test is not None else None,
            'transactions_shape': self.transactions.shape if self.transactions is not None else None,
            'unique_routes_train': len(set(zip(self.train['srcid'], self.train['destid']))) if self.train is not None else None,
            'unique_routes_test': len(set(zip(self.test['srcid'], self.test['destid']))) if self.test is not None else None,
            'target_stats': self.train['final_seatcount'].describe().to_dict() if self.train is not None else None
        }
        
        return summary
    
    def save_processed_data(self, output_dir: Path) -> None:
        """Save processed data to specified directory"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.train is not None:
            self.train.to_csv(output_dir / "train_processed.csv", index=False)
        if self.test is not None:
            self.test.to_csv(output_dir / "test_processed.csv", index=False)
        if self.transactions is not None:
            self.transactions.to_csv(output_dir / "transactions_processed.csv", index=False)
        
        logger.info(f"Processed data saved to {output_dir}")

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience function to load all data"""
    loader = DataLoader()
    return loader.load_all_data()