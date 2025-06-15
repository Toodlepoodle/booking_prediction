"""
Feature engineering module for bus demand forecasting
"""

import pandas as pd
import numpy as np
import holidays
from typing import List, Tuple, Dict
import logging
from .config import FEATURE_CONFIG

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Class for comprehensive feature engineering"""
    
    def __init__(self):
        self.feature_config = FEATURE_CONFIG
        self.india_holidays = holidays.India(years=range(2022, 2026))
        
    def create_calendar_features(self, df: pd.DataFrame, date_col: str = 'doj') -> pd.DataFrame:
        """Create comprehensive calendar-based features"""
        df = df.copy()
        
        logger.info(f"Creating calendar features for {date_col}")
        
        # Basic date features
        df[f'{date_col}_year'] = df[date_col].dt.year
        df[f'{date_col}_month'] = df[date_col].dt.month
        df[f'{date_col}_day'] = df[date_col].dt.day
        df[f'{date_col}_dayofweek'] = df[date_col].dt.dayofweek
        df[f'{date_col}_dayofyear'] = df[date_col].dt.dayofyear
        df[f'{date_col}_weekofyear'] = df[date_col].dt.isocalendar().week
        df[f'{date_col}_quarter'] = df[date_col].dt.quarter
        
        # Weekend and special day indicators
        df[f'{date_col}_is_weekend'] = (df[f'{date_col}_dayofweek'] >= 5).astype(int)
        df[f'{date_col}_is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        df[f'{date_col}_is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        
        # Season encoding
        df[f'{date_col}_season'] = df[f'{date_col}_month'].apply(self._get_season)
        
        # Cyclical encoding for time features
        df[f'{date_col}_month_sin'] = np.sin(2 * np.pi * df[f'{date_col}_month'] / 12)
        df[f'{date_col}_month_cos'] = np.cos(2 * np.pi * df[f'{date_col}_month'] / 12)
        df[f'{date_col}_day_sin'] = np.sin(2 * np.pi * df[f'{date_col}_day'] / 31)
        df[f'{date_col}_day_cos'] = np.cos(2 * np.pi * df[f'{date_col}_day'] / 31)
        df[f'{date_col}_dayofweek_sin'] = np.sin(2 * np.pi * df[f'{date_col}_dayofweek'] / 7)
        df[f'{date_col}_dayofweek_cos'] = np.cos(2 * np.pi * df[f'{date_col}_dayofweek'] / 7)
        
        return df
    
    def create_holiday_features(self, df: pd.DataFrame, date_col: str = 'doj') -> pd.DataFrame:
        """Create holiday-related features"""
        df = df.copy()
        
        logger.info(f"Creating holiday features for {date_col}")
        
        # Holiday indicator - fix date comparison
        df[f'{date_col}_is_holiday'] = df[date_col].apply(
            lambda x: x.date() in self.india_holidays
        ).astype(int)
        
        # Days to/from nearest holiday
        df[f'{date_col}_days_to_holiday'] = df[date_col].apply(self._days_to_holiday)
        df[f'{date_col}_days_from_holiday'] = df[date_col].apply(self._days_from_holiday)
        
        # Near holiday indicator
        proximity_days = self.feature_config['holiday_proximity_days']
        df[f'{date_col}_near_holiday'] = (
            (df[f'{date_col}_days_to_holiday'] <= proximity_days) | 
            (df[f'{date_col}_days_from_holiday'] <= proximity_days)
        ).astype(int)
        
        return df
    
    def create_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create route-specific features"""
        df = df.copy()
        
        logger.info("Creating route features")
        
        # Route identifiers
        df['route'] = df['srcid'].astype(str) + '_' + df['destid'].astype(str)
        df['reverse_route'] = df['destid'].astype(str) + '_' + df['srcid'].astype(str)
        
        # Same region indicator
        if 'srcid_region' in df.columns and 'destid_region' in df.columns:
            df['same_region'] = (df['srcid_region'] == df['destid_region']).astype(int)
            
        # Tier-based features
        if 'srcid_tier' in df.columns and 'destid_tier' in df.columns:
            df['tier_combination'] = df['srcid_tier'] + '_to_' + df['destid_tier']
            df['tier_downgrade'] = (df['srcid_tier'] < df['destid_tier']).astype(int)
            df['tier_upgrade'] = (df['srcid_tier'] > df['destid_tier']).astype(int)
            df['same_tier'] = (df['srcid_tier'] == df['destid_tier']).astype(int)
        
        return df
    
    def create_booking_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on booking patterns"""
        df = df.copy()
        
        logger.info("Creating booking pattern features")
        
        # Booking velocity and conversion metrics
        df['booking_velocity'] = df['cumsum_seatcount'] / (df['cumsum_searchcount'] + 1)
        df['conversion_rate'] = df['cumsum_seatcount'] / (df['cumsum_searchcount'] + 1)
        
        # Booking timing features
        if 'dbd' in df.columns:
            df['early_booking'] = (df['dbd'] > 20).astype(int)
            df['last_minute_booking'] = (df['dbd'] < 7).astype(int)
            df['medium_advance_booking'] = ((df['dbd'] >= 7) & (df['dbd'] <= 20)).astype(int)
        
        return df
    
    def create_aggregated_features(self, transactions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create aggregated features at different levels"""
        logger.info("Creating aggregated features")
        
        # Route-level aggregations
        route_agg = transactions_df.groupby(['srcid', 'destid']).agg({
            'cumsum_seatcount': ['mean', 'max', 'std', 'median', 'sum'],
            'cumsum_searchcount': ['mean', 'max', 'std', 'median', 'sum'],
            'dbd': ['mean', 'max', 'min', 'std']
        }).reset_index()
        
        route_agg.columns = ['srcid', 'destid'] + [
            f'route_{col[0]}_{col[1]}' for col in route_agg.columns[2:]
        ]
        
        # Add route popularity metrics
        route_agg['route_booking_frequency'] = transactions_df.groupby(['srcid', 'destid']).size().values
        route_agg['route_avg_conversion'] = route_agg['route_cumsum_seatcount_mean'] / (route_agg['route_cumsum_searchcount_mean'] + 1)
        
        # Source city aggregations
        src_agg = transactions_df.groupby('srcid').agg({
            'cumsum_seatcount': ['mean', 'sum', 'std'],
            'cumsum_searchcount': ['mean', 'sum', 'std']
        }).reset_index()
        
        src_agg.columns = ['srcid'] + [f'src_{col[0]}_{col[1]}' for col in src_agg.columns[1:]]
        
        # Destination city aggregations
        dest_agg = transactions_df.groupby('destid').agg({
            'cumsum_seatcount': ['mean', 'sum', 'std'],
            'cumsum_searchcount': ['mean', 'sum', 'std']
        }).reset_index()
        
        dest_agg.columns = ['destid'] + [f'dest_{col[0]}_{col[1]}' for col in dest_agg.columns[1:]]
        
        return route_agg, src_agg, dest_agg
    
    def create_time_series_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Create time series and lag features"""
        logger.info("Creating time series features")
        
        df = transactions_df.copy()
        df = df.sort_values(['srcid', 'destid', 'doj', 'doi'])
        
        # Group by route
        route_groups = df.groupby(['srcid', 'destid'])
        
        lag_features = []
        lag_periods = self.feature_config['lag_periods']
        rolling_windows = self.feature_config['rolling_windows']
        
        for name, group in route_groups:
            group_sorted = group.sort_values(['doj', 'doi'])
            
            # Lag features
            for lag in lag_periods:
                group_sorted[f'seatcount_lag_{lag}'] = group_sorted['cumsum_seatcount'].shift(lag)
                group_sorted[f'searchcount_lag_{lag}'] = group_sorted['cumsum_searchcount'].shift(lag)
            
            # Rolling statistics
            for window in rolling_windows:
                group_sorted[f'seatcount_roll_mean_{window}'] = (
                    group_sorted['cumsum_seatcount'].rolling(window=window, min_periods=1).mean()
                )
                group_sorted[f'searchcount_roll_mean_{window}'] = (
                    group_sorted['cumsum_searchcount'].rolling(window=window, min_periods=1).mean()
                )
                group_sorted[f'seatcount_roll_std_{window}'] = (
                    group_sorted['cumsum_seatcount'].rolling(window=window, min_periods=1).std()
                )
                group_sorted[f'searchcount_roll_std_{window}'] = (
                    group_sorted['cumsum_searchcount'].rolling(window=window, min_periods=1).std()
                )
            
            lag_features.append(group_sorted)
        
        return pd.concat(lag_features, ignore_index=True)
    
    def create_all_features(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame, 
        transactions_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create all features for train and test sets"""
        logger.info("Starting comprehensive feature engineering")
        
        # Filter transactions for prediction window
        dbd_filter = self.feature_config['dbd_filter']
        transactions_15d = transactions_df[transactions_df['dbd'] == dbd_filter].copy()
        
        # Create features for each dataset
        datasets = {'train': train_df.copy(), 'test': test_df.copy(), 'transactions': transactions_15d.copy()}
        
        for name, df in datasets.items():
            logger.info(f"Processing {name} dataset")
            
            # Calendar features
            df = self.create_calendar_features(df)
            
            # Holiday features
            df = self.create_holiday_features(df)
            
            # Route features
            df = self.create_route_features(df)
            
            # Booking pattern features (only for transactions)
            if name == 'transactions':
                df = self.create_booking_pattern_features(df)
            
            datasets[name] = df
        
        # Create aggregated features from full transactions
        route_agg, src_agg, dest_agg = self.create_aggregated_features(transactions_df)
        
        # Merge aggregated features
        for name in ['train', 'test']:
            datasets[name] = datasets[name].merge(route_agg, on=['srcid', 'destid'], how='left')
            datasets[name] = datasets[name].merge(src_agg, on='srcid', how='left')
            datasets[name] = datasets[name].merge(dest_agg, on='destid', how='left')
        
        # Create and merge 15-day prediction features
        transactions_features = self._create_prediction_features(datasets['transactions'])
        
        datasets['train'] = datasets['train'].merge(
            transactions_features, on=['srcid', 'destid', 'doj'], how='left'
        )
        datasets['test'] = datasets['test'].merge(
            transactions_features, on=['srcid', 'destid', 'doj'], how='left'
        )
        
        # Handle missing values
        datasets['train'] = self._handle_missing_values(datasets['train'])
        datasets['test'] = self._handle_missing_values(datasets['test'], reference_df=datasets['train'])
        
        logger.info("Feature engineering completed!")
        logger.info(f"Final train shape: {datasets['train'].shape}")
        logger.info(f"Final test shape: {datasets['test'].shape}")
        
        return datasets['train'], datasets['test']
    
    def _create_prediction_features(self, transactions_15d: pd.DataFrame) -> pd.DataFrame:
        """Create features specifically for 15-day prediction window"""
        features = transactions_15d.groupby(['srcid', 'destid', 'doj']).agg({
            'cumsum_seatcount': 'max',
            'cumsum_searchcount': 'max',
            'booking_velocity': 'mean',
            'conversion_rate': 'mean'
        }).reset_index()
        
        features.rename(columns={
            'cumsum_seatcount': 'seats_15d_before',
            'cumsum_searchcount': 'searches_15d_before',
            'booking_velocity': 'velocity_15d_before',
            'conversion_rate': 'conversion_15d_before'
        }, inplace=True)
        
        return features
    
    def _handle_missing_values(self, df: pd.DataFrame, reference_df: pd.DataFrame = None) -> pd.DataFrame:
        """Handle missing values in features"""
        df = df.copy()
        
        # Use reference dataframe for imputation values if provided
        ref_df = reference_df if reference_df is not None else df
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['final_seatcount']  # Don't impute target variable
        
        for col in numeric_columns:
            if col not in exclude_cols:
                fill_value = ref_df[col].median()
                df[col].fillna(fill_value, inplace=True)
        
        return df
    
    def _get_season(self, month: int) -> int:
        """Get season based on month"""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Autumn
    
    def _days_to_holiday(self, date) -> int:
        """Calculate days to next holiday"""
        date_only = date.date() if hasattr(date, 'date') else date
        upcoming_holidays = [h for h in self.india_holidays if h > date_only]
        if upcoming_holidays:
            return (min(upcoming_holidays) - date_only).days
        return 365
    
    def _days_from_holiday(self, date) -> int:
        """Calculate days from previous holiday"""
        date_only = date.date() if hasattr(date, 'date') else date
        past_holidays = [h for h in self.india_holidays if h < date_only]
        if past_holidays:
            return (date_only - max(past_holidays)).days
        return 365