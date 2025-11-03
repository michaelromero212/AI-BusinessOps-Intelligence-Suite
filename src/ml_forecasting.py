"""
Machine learning forecasting module for the AI Business Operations Intelligence Suite.
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Union
import logging
import joblib
import os

logger = logging.getLogger(__name__)

class MLForecaster:
    """Handles machine learning model training and predictions for different business scenarios."""
    
    def __init__(self, dataset_type: str = "hr"):
        """
        Initialize the MLForecaster.
        
        Args:
            dataset_type (str): Type of dataset to process ('hr', 'procurement', or 'finance')
        """
        self.dataset_type = dataset_type.lower()
        self.model = None
        self.feature_columns = []
        self.target_column = None
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for model training.
        
        Args:
            df (pd.DataFrame): Input dataset with engineered features
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target variables
        """
        if self.dataset_type == 'hr':
            return self._prepare_hr_data(df)
        elif self.dataset_type == 'procurement':
            return self._prepare_procurement_data(df)
        elif self.dataset_type == 'finance':
            return self._prepare_finance_data(df)
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train the appropriate model based on dataset type.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            Dict: Model performance metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if self.dataset_type == 'hr':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            metrics = self._train_classifier(X_train, X_test, y_train, y_test)
        
        elif self.dataset_type == 'procurement':
            self.model = LinearRegression()
            metrics = self._train_regressor(X_train, X_test, y_train, y_test)
        
        elif self.dataset_type == 'finance':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            metrics = self._train_regressor(X_train, X_test, y_train, y_test)
            
        logger.info(f"Model training completed for {self.dataset_type} dataset")
        return metrics

    def predict(self, X: pd.DataFrame) -> Union[np.ndarray, pd.Series]:
        """
        Generate predictions using the trained model.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            Union[np.ndarray, pd.Series]: Model predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
            
        return self.model.predict(X)

    def _prepare_hr_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare HR data for attrition prediction."""
        # Select relevant features
        feature_cols = [
            'tenure_ratio', 'company_change_rate', 'income_per_year_service',
            'overtime_flag', 'dept_attrition_rate'
        ]
        
        # Add any numerical columns from original data
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        feature_cols.extend([col for col in numerical_cols if col not in feature_cols + ['attrition']])
        
        self.feature_columns = feature_cols
        self.target_column = 'attrition'
        
        # Convert target to binary
        if df[self.target_column].dtype == 'object':
            df[self.target_column] = (df[self.target_column] == 'Yes').astype(int)
        
        return df[self.feature_columns], df[self.target_column]

    def _prepare_procurement_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare procurement data for spend forecasting."""
        feature_cols = [
            'avg_spend', 'spend_std', 'transaction_count',
            'spend_concentration', 'on_time_delivery'
        ]
        
        self.feature_columns = feature_cols
        self.target_column = 'spend_amount'
        
        return df[self.feature_columns], df[self.target_column]

    def _prepare_finance_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare finance data for performance forecasting."""
        feature_cols = [
            'revenue_growth', 'expense_growth',
            'revenue_ma_2Q', 'revenue_ma_4Q',
            'profit_ma_2Q', 'profit_ma_4Q'
        ]
        
        self.feature_columns = feature_cols
        self.target_column = 'profit'
        
        return df[self.feature_columns], df[self.target_column]

    def _train_classifier(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                         y_train: pd.Series, y_test: pd.Series) -> Dict:
        """Train and evaluate a classification model."""
        self.model.fit(X_train, y_train)
        
        # Generate predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        return metrics

    def _train_regressor(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                        y_train: pd.Series, y_test: pd.Series) -> Dict:
        """Train and evaluate a regression model."""
        self.model.fit(X_train, y_train)
        
        # Generate predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': self.model.score(X_test, y_test)
        }
        
        return metrics

    def save_model(self, path: str):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load a trained model from disk."""
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")

if __name__ == "__main__":
    # Example usage
    from src.data_loader import DataLoader
    from src.feature_engineering import FeatureEngineer
    
    # Load and prepare data
    loader = DataLoader(dataset_type="hr")
    df = loader.load_data()
    
    # Create features
    engineer = FeatureEngineer(dataset_type="hr")
    df_featured = engineer.create_features(df)
    
    # Train model
    forecaster = MLForecaster(dataset_type="hr")
    X, y = forecaster.prepare_data(df_featured)
    metrics = forecaster.train_model(X, y)
    
    print("\nModel Performance Metrics:")
    print(metrics)