"""
Business insights generation module.
"""

import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class InsightGenerator:
    """Generates narrative insights from business data."""
    
    def __init__(self):
        """Initialize the InsightGenerator."""
        logger.info("Initializing insight generator...")

import torch
from transformers import pipeline
import pandas as pd
from typing import Dict, List
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class InsightGenerator:
    """Generates narrative insights from business data using Hugging Face models."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the InsightGenerator.
        
        Args:
            model_name (str): Name of the Hugging Face model to use
        """
        self.model_name = model_name or os.getenv('HF_MODEL_NAME', 'facebook/bart-large-cnn')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Loading model {self.model_name}...")
        self.summarizer = pipeline(
            "summarization",
            model=self.model_name,
            device=-1  # Use CPU
        )
        
    def generate_insights(self, metrics: Dict, predictions: pd.DataFrame, 
                        dataset_type: str) -> str:
        """
        Generate narrative insights from model metrics and predictions.
        
        Args:
            metrics (Dict): Model performance metrics
            predictions (pd.DataFrame): Model predictions
            dataset_type (str): Type of dataset ('hr', 'procurement', or 'finance')
            
        Returns:
            str: Generated narrative insights
        """
        # Create input text based on dataset type
        input_text = self._prepare_input_text(metrics, predictions, dataset_type)
        
        # Generate summary
        summary = self._generate_summary(input_text)
        
        return summary

    def _prepare_input_text(self, metrics: Dict, predictions: pd.DataFrame, 
                          dataset_type: str) -> str:
        """Prepare input text for the model based on dataset type."""
        text_parts = []
        
        if dataset_type == 'hr':
            text_parts.extend(self._prepare_hr_insights(metrics, predictions))
        elif dataset_type == 'procurement':
            text_parts.extend(self._prepare_procurement_insights(metrics, predictions))
        elif dataset_type == 'finance':
            text_parts.extend(self._prepare_finance_insights(metrics, predictions))
        
        return " ".join(text_parts)

    def _prepare_hr_insights(self, metrics: Dict, predictions: pd.DataFrame) -> List[str]:
        """Prepare HR-specific insight text."""
        insights = []
        
        # Model performance insights
        if 'accuracy' in metrics:
            insights.append(f"The attrition prediction model achieved {metrics['accuracy']:.2%} accuracy.")
        
        # Attrition risk insights
        high_risk = predictions[predictions['attrition_probability'] > 0.7]
        insights.append(f"Identified {len(high_risk)} employees at high risk of attrition.")
        
        if not high_risk.empty:
            dept_risk = high_risk['department'].value_counts()
            highest_risk_dept = dept_risk.index[0]
            insights.append(f"The {highest_risk_dept} department shows the highest attrition risk.")
        
        return insights

    def _prepare_procurement_insights(self, metrics: Dict, predictions: pd.DataFrame) -> List[str]:
        """Prepare procurement-specific insight text."""
        insights = []
        
        # Model performance insights
        if 'rmse' in metrics:
            insights.append(f"The spend forecasting model achieved RMSE of ${metrics['rmse']:,.2f}.")
        
        # Spend analysis insights
        if 'predicted_spend' in predictions.columns:
            total_predicted = predictions['predicted_spend'].sum()
            insights.append(f"Total predicted spend for next quarter: ${total_predicted:,.2f}")
            
            # Identify unusual patterns
            unusual_spend = predictions[predictions['spend_variance'] > 2]
            if not unusual_spend.empty:
                insights.append(f"Identified {len(unusual_spend)} vendors with unusual spending patterns.")
        
        return insights

    def _prepare_finance_insights(self, metrics: Dict, predictions: pd.DataFrame) -> List[str]:
        """Prepare finance-specific insight text."""
        insights = []
        
        # Model performance insights
        if 'r2' in metrics:
            insights.append(f"The financial forecasting model explains {metrics['r2']:.2%} of profit variance.")
        
        # Financial performance insights
        if 'predicted_profit' in predictions.columns:
            avg_profit = predictions['predicted_profit'].mean()
            profit_trend = 'positive' if avg_profit > 0 else 'negative'
            insights.append(f"Average predicted profit trend is {profit_trend} at ${abs(avg_profit):,.2f}.")
            
            # Identify potential issues
            loss_risk = predictions[predictions['predicted_profit'] < 0]
            if not loss_risk.empty:
                insights.append(f"Warning: {len(loss_risk)} periods show risk of losses.")
        
        return insights

    def _generate_summary(self, input_text: str) -> str:
        """Generate a concise summary from the input text."""
        insights = [line for line in input_text.split('\n') if line.strip()]
        return '\n'.join(insights)

if __name__ == "__main__":
    # Example usage
    from src.data_loader import DataLoader
    from src.feature_engineering import FeatureEngineer
    from src.ml_forecasting import MLForecaster
    
    # Load and process data
    loader = DataLoader(dataset_type="hr")
    df = loader.load_data()
    
    engineer = FeatureEngineer(dataset_type="hr")
    df_featured = engineer.create_features(df)
    
    # Train model and get predictions
    forecaster = MLForecaster(dataset_type="hr")
    X, y = forecaster.prepare_data(df_featured)
    metrics = forecaster.train_model(X, y)
    
    predictions = pd.DataFrame({
        'attrition_probability': forecaster.predict(X),
        'department': df_featured['department']
    })
    
    # Generate insights
    generator = InsightGenerator()
    insights = generator.generate_insights(metrics, predictions, dataset_type="hr")
    
    print("\nGenerated Insights:")
    print(insights)