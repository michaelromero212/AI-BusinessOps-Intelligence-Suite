"""
Data loading and preprocessing module for the AI Business Operations Intelligence Suite.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DataLoader:
    """Handles data loading and preprocessing for various business datasets."""
    
    def __init__(self, dataset_type: str = "hr"):
        """
        Initialize the DataLoader.
        
        Args:
            dataset_type (str): Type of dataset to load ('hr', 'procurement', or 'finance')
        """
        self.dataset_type = dataset_type.lower()
        self.data_path = os.getenv('LOCAL_DATA_PATH')
        
        if not self.data_path:
            raise ValueError("LOCAL_DATA_PATH not set in environment variables")
            
        self.dataset_mappings = {
            'hr': 'hr_attrition.csv',
            'procurement': 'procurement.csv',
            'finance': 'finance.csv'
        }

    def load_data(self) -> pd.DataFrame:
        """
        Load the specified dataset and perform initial preprocessing.
        
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        if self.dataset_type not in self.dataset_mappings:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
            
        file_path = os.path.join(self.data_path, self.dataset_mappings[self.dataset_type])
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {self.dataset_type} dataset with {len(df)} records")
            return self._preprocess_data(df)
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {file_path}")
            raise

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform dataset-specific preprocessing.
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        if self.dataset_type == 'hr':
            return self._preprocess_hr_data(df)
        elif self.dataset_type == 'procurement':
            return self._preprocess_procurement_data(df)
        elif self.dataset_type == 'finance':
            return self._preprocess_finance_data(df)
        
        return df

    def _preprocess_hr_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess HR attrition dataset."""
        # Handle missing values
        df = df.fillna({
            'TotalWorkingYears': 0,
            'NumCompaniesWorked': 0,
            'YearsInCurrentRole': 0
        })
        
        # Create standardized column names
        df = df.rename(columns={
            'EmployeeNumber': 'employee_id',
            'Attrition': 'attrition',
            'Department': 'department',
            'MonthlyIncome': 'monthly_income'
        })
        
        return df

    def _preprocess_procurement_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess procurement dataset."""
        # Handle missing values and standardize column names
        df = df.fillna({
            'Spend_Amount': 0,
            'Delivery_Days': 0
        })
        
        df = df.rename(columns={
            'Vendor_ID': 'vendor_id',
            'Spend_Amount': 'spend_amount',
            'Category': 'category'
        })
        
        return df

    def _preprocess_finance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess finance dataset."""
        # Handle missing values and standardize column names
        df = df.fillna({
            'Revenue': 0,
            'Expenses': 0
        })
        
        df = df.rename(columns={
            'Quarter': 'quarter',
            'Revenue': 'revenue',
            'Expenses': 'expenses'
        })
        
        return df

if __name__ == "__main__":
    # Example usage
    loader = DataLoader(dataset_type="hr")
    df = loader.load_data()
    print(f"Loaded dataset shape: {df.shape}")
    print("\nSample data:")
    print(df.head())