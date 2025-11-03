"""
Feature engineering module for the AI Business Operations Intelligence Suite.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Handles feature engineering for different types of business data."""
    
    def __init__(self, dataset_type: str = "hr"):
        """
        Initialize the FeatureEngineer.
        
        Args:
            dataset_type (str): Type of dataset to process ('hr', 'procurement', or 'finance')
        """
        self.dataset_type = dataset_type.lower()

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create dataset-specific features.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with engineered features
        """
        if self.dataset_type == 'hr':
            return self._create_hr_features(df)
        elif self.dataset_type == 'procurement':
            return self._create_procurement_features(df)
        elif self.dataset_type == 'finance':
            return self._create_finance_features(df)
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def _create_hr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for HR attrition analysis."""
        # Create copy to avoid modifying original
        df_features = df.copy()
        
        # Calculate tenure-related features
        df_features['tenure_ratio'] = df_features['YearsInCurrentRole'] / (df_features['TotalWorkingYears'] + 1)
        df_features['company_change_rate'] = df_features['NumCompaniesWorked'] / (df_features['TotalWorkingYears'] + 1)
        
        # Calculate compensation features
        if 'monthly_income' in df_features.columns:
            df_features['income_per_year_service'] = df_features['monthly_income'] * 12 / (df_features['TotalWorkingYears'] + 1)
        
        # Create work-life balance indicators
        if 'OverTime' in df_features.columns:
            df_features['overtime_flag'] = (df_features['OverTime'] == 'Yes').astype(int)
        
        # Department-based features
        if 'department' in df_features.columns:
            dept_attrition = df_features.groupby('department')['attrition'].mean()
            df_features['dept_attrition_rate'] = df_features['department'].map(dept_attrition)
        
        logger.info("Created HR features: tenure_ratio, company_change_rate, etc.")
        return df_features

    def _create_procurement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for procurement analysis."""
        df_features = df.copy()
        
        # Vendor performance metrics
        if all(col in df_features.columns for col in ['vendor_id', 'spend_amount']):
            # Calculate vendor spending patterns
            vendor_stats = df_features.groupby('vendor_id').agg({
                'spend_amount': ['mean', 'std', 'count']
            }).reset_index()
            
            vendor_stats.columns = ['vendor_id', 'avg_spend', 'spend_std', 'transaction_count']
            df_features = df_features.merge(vendor_stats, on='vendor_id', how='left')
            
            # Calculate spend concentration
            df_features['spend_concentration'] = df_features['spend_amount'] / df_features['avg_spend']
        
        # Delivery performance
        if 'Delivery_Days' in df_features.columns:
            df_features['delivery_delay'] = df_features['Delivery_Days'] - df_features['Expected_Delivery_Days']
            df_features['on_time_delivery'] = (df_features['delivery_delay'] <= 0).astype(int)
        
        logger.info("Created procurement features: spend_concentration, delivery_delay, etc.")
        return df_features

    def _create_finance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for financial analysis."""
        df_features = df.copy()
        
        # Basic financial ratios
        if all(col in df_features.columns for col in ['revenue', 'expenses']):
            df_features['profit'] = df_features['revenue'] - df_features['expenses']
            df_features['profit_margin'] = df_features['profit'] / df_features['revenue']
            
            # Rolling metrics if date information is available
            if 'quarter' in df_features.columns:
                df_features = df_features.sort_values('quarter')
                df_features['revenue_growth'] = df_features['revenue'].pct_change()
                df_features['expense_growth'] = df_features['expenses'].pct_change()
                
                # Rolling averages
                for window in [2, 4]:  # 2 quarters and annual
                    df_features[f'revenue_ma_{window}Q'] = df_features['revenue'].rolling(window).mean()
                    df_features[f'profit_ma_{window}Q'] = df_features['profit'].rolling(window).mean()
        
        logger.info("Created finance features: profit_margin, revenue_growth, etc.")
        return df_features

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from src.data_loader import DataLoader
    
    # Load data
    loader = DataLoader(dataset_type="hr")
    df = loader.load_data()
    
    # Create features
    engineer = FeatureEngineer(dataset_type="hr")
    df_featured = engineer.create_features(df)
    
    print("Original columns:", df.columns.tolist())
    print("\nNew columns:", [col for col in df_featured.columns if col not in df.columns])