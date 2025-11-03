"""
Dashboard creation module using Streamlit for visualizing business insights.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Dashboard:
    """Creates interactive dashboards for visualizing business insights."""
    
    def __init__(self, dataset_type: str = "hr"):
        """
        Initialize the Dashboard.
        
        Args:
            dataset_type (str): Type of dataset to visualize ('hr', 'procurement', or 'finance')
        """
        self.dataset_type = dataset_type.lower()
        
    def create_dashboard(self, data: pd.DataFrame, predictions: pd.DataFrame, metrics: Dict):
        """
        Create and display the Streamlit dashboard.
        
        Args:
            data (pd.DataFrame): Original dataset with features
            predictions (pd.DataFrame): Model predictions
            metrics (Dict): Model performance metrics
        """
        st.title(f"Business Operations Intelligence Dashboard - {self.dataset_type.upper()}")
        
        # Display metrics summary
        self._display_metrics_summary(metrics)
        
        # Dataset-specific visualizations
        if self.dataset_type == 'hr':
            self._create_hr_dashboard(data, predictions)
        elif self.dataset_type == 'procurement':
            self._create_procurement_dashboard(data, predictions)
        elif self.dataset_type == 'finance':
            self._create_finance_dashboard(data, predictions)

    def _display_metrics_summary(self, metrics: Dict):
        """Display model performance metrics."""
        st.header("Model Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        if self.dataset_type == 'hr':
            if 'accuracy' in metrics:
                col1.metric("Model Accuracy", f"{metrics['accuracy']:.2%}")
            if 'classification_report' in metrics:
                st.text("Detailed Classification Report:")
                st.text(metrics['classification_report'])
                
        else:  # For regression models (procurement and finance)
            if 'rmse' in metrics:
                col1.metric("RMSE", f"${metrics['rmse']:,.2f}")
            if 'r2' in metrics:
                col2.metric("R² Score", f"{metrics['r2']:.2%}")
            if 'mse' in metrics:
                col3.metric("MSE", f"${metrics['mse']:,.2f}")

    def _create_hr_dashboard(self, data: pd.DataFrame, predictions: pd.DataFrame):
        """Create HR-specific visualizations."""
        st.header("HR Analytics Dashboard")
        
        # Attrition Risk Distribution
        st.subheader("Attrition Risk Distribution")
        fig = px.histogram(predictions, x='attrition_probability', 
                         title="Distribution of Attrition Risk Scores",
                         nbins=30)
        st.plotly_chart(fig)
        
        # Department-wise Attrition Risk
        if 'department' in data.columns:
            st.subheader("Department-wise Attrition Risk")
            dept_risk = predictions.groupby('department')['attrition_probability'].mean()
            fig = px.bar(dept_risk, title="Average Attrition Risk by Department")
            st.plotly_chart(fig)
        
        # High Risk Employees
        st.subheader("High Risk Employees")
        high_risk = predictions[predictions['attrition_probability'] > 0.7]
        if not high_risk.empty:
            st.write(f"Number of high-risk employees: {len(high_risk)}")
            st.dataframe(high_risk)

    def _create_procurement_dashboard(self, data: pd.DataFrame, predictions: pd.DataFrame):
        """Create procurement-specific visualizations."""
        st.header("Procurement Analytics Dashboard")
        
        # Spend Forecasts
        st.subheader("Spend Forecasts")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['spend_amount'],
                               name="Actual Spend"))
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions['predicted_spend'],
                               name="Predicted Spend"))
        st.plotly_chart(fig)
        
        # Vendor Analysis
        if 'vendor_id' in data.columns:
            st.subheader("Top Vendors by Spend")
            vendor_spend = data.groupby('vendor_id')['spend_amount'].sum().sort_values(ascending=False)
            fig = px.bar(vendor_spend.head(10), title="Top 10 Vendors by Total Spend")
            st.plotly_chart(fig)
        
        # Spend Anomalies
        st.subheader("Spend Anomalies")
        anomalies = predictions[predictions['spend_variance'] > 2]
        if not anomalies.empty:
            st.write(f"Number of anomalous transactions: {len(anomalies)}")
            st.dataframe(anomalies)

    def _create_finance_dashboard(self, data: pd.DataFrame, predictions: pd.DataFrame):
        """Create finance-specific visualizations."""
        st.header("Financial Analytics Dashboard")
        
        # Profit Trends
        st.subheader("Profit Trends")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['profit'],
                               name="Actual Profit"))
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions['predicted_profit'],
                               name="Predicted Profit"))
        st.plotly_chart(fig)
        
        # Revenue vs Expenses
        if all(col in data.columns for col in ['revenue', 'expenses']):
            st.subheader("Revenue vs Expenses")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=data.index, y=data['revenue'],
                               name="Revenue"))
            fig.add_trace(go.Bar(x=data.index, y=data['expenses'],
                               name="Expenses"))
            st.plotly_chart(fig)
        
        # Profit Margin Analysis
        if 'profit_margin' in data.columns:
            st.subheader("Profit Margin Analysis")
            fig = px.line(data, x=data.index, y='profit_margin',
                         title="Profit Margin Over Time")
            st.plotly_chart(fig)

    def save_visualizations(self, data: pd.DataFrame, predictions: pd.DataFrame, 
                          output_dir: str = "outputs/visualizations"):
        """
        Save static versions of the visualizations.
        
        Args:
            data (pd.DataFrame): Original dataset with features
            predictions (pd.DataFrame): Model predictions
            output_dir (str): Directory to save visualizations
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if self.dataset_type == 'hr':
            self._save_hr_visualizations(data, predictions, output_dir)
        elif self.dataset_type == 'procurement':
            self._save_procurement_visualizations(data, predictions, output_dir)
        elif self.dataset_type == 'finance':
            self._save_finance_visualizations(data, predictions, output_dir)
            
        logger.info(f"Visualizations saved to {output_dir}")

    def _save_hr_visualizations(self, data: pd.DataFrame, predictions: pd.DataFrame, 
                              output_dir: str):
        """Save HR-specific visualizations."""
        # Attrition Risk Distribution
        fig = px.histogram(predictions, x='attrition_probability',
                         title="Distribution of Attrition Risk Scores")
        fig.write_html(f"{output_dir}/attrition_risk_distribution.html")
        
        # Department-wise Risk
        if 'department' in data.columns:
            dept_risk = predictions.groupby('department')['attrition_probability'].mean()
            fig = px.bar(dept_risk, title="Average Attrition Risk by Department")
            fig.write_html(f"{output_dir}/department_risk.html")

    def _save_procurement_visualizations(self, data: pd.DataFrame, predictions: pd.DataFrame, 
                                      output_dir: str):
        """Save procurement-specific visualizations."""
        # Spend Forecasts
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['spend_amount'],
                               name="Actual Spend"))
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions['predicted_spend'],
                               name="Predicted Spend"))
        fig.write_html(f"{output_dir}/spend_forecast.html")
        
        # Vendor Analysis
        if 'vendor_id' in data.columns:
            vendor_spend = data.groupby('vendor_id')['spend_amount'].sum()
            fig = px.bar(vendor_spend.head(10), title="Top 10 Vendors by Total Spend")
            fig.write_html(f"{output_dir}/top_vendors.html")

    def _save_finance_visualizations(self, data: pd.DataFrame, predictions: pd.DataFrame, 
                                   output_dir: str):
        """Save finance-specific visualizations."""
        # Profit Trends
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['profit'],
                               name="Actual Profit"))
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions['predicted_profit'],
                               name="Predicted Profit"))
        fig.write_html(f"{output_dir}/profit_forecast.html")
        
        # Revenue vs Expenses
        if all(col in data.columns for col in ['revenue', 'expenses']):
            fig = go.Figure()
            fig.add_trace(go.Bar(x=data.index, y=data['revenue'],
                               name="Revenue"))
            fig.add_trace(go.Bar(x=data.index, y=data['expenses'],
                               name="Expenses"))
            fig.write_html(f"{output_dir}/revenue_expenses.html")

if __name__ == "__main__":
    # Example usage
    import streamlit as st
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
    
    # Create dashboard
    dashboard = Dashboard(dataset_type="hr")
    dashboard.create_dashboard(df_featured, predictions, metrics)
    
    # Save static visualizations
    dashboard.save_visualizations(df_featured, predictions)