"""
Pipeline module for orchestrating the end-to-end business intelligence workflow.
"""

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.ml_forecasting import MLForecaster
from src.hf_insights import InsightGenerator
from src.dashboard import Dashboard
import pandas as pd
from typing import Dict, Optional
import logging
from tqdm import tqdm
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Pipeline:
    """Orchestrates the end-to-end business intelligence workflow."""
    
    def __init__(self, dataset_type: str = "hr", output_dir: str = "outputs"):
        """
        Initialize the Pipeline.
        
        Args:
            dataset_type (str): Type of dataset to process ('hr', 'procurement', or 'finance')
            output_dir (str): Directory to save outputs
        """
        self.dataset_type = dataset_type.lower()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader(dataset_type=dataset_type)
        self.feature_engineer = FeatureEngineer(dataset_type=dataset_type)
        self.forecaster = MLForecaster(dataset_type=dataset_type)
        self.insight_generator = InsightGenerator()
        self.dashboard = Dashboard(dataset_type=dataset_type)
        
        # Storage for intermediate results
        self.data = None
        self.featured_data = None
        self.predictions = None
        self.metrics = None
        self.insights = None

    def run(self, save_artifacts: bool = True) -> Dict:
        """
        Run the complete pipeline.
        
        Args:
            save_artifacts (bool): Whether to save intermediate outputs
            
        Returns:
            Dict: Pipeline results including metrics and file paths
        """
        logger.info("Starting pipeline execution...")
        results = {}
        
        try:
            # 1. Load Data
            with tqdm(desc="Loading data", total=1) as pbar:
                self.data = self.data_loader.load_data()
                pbar.update(1)
                
            # 2. Feature Engineering
            with tqdm(desc="Engineering features", total=1) as pbar:
                self.featured_data = self.feature_engineer.create_features(self.data)
                pbar.update(1)
                
            # 3. Train Model and Generate Predictions
            with tqdm(desc="Training model", total=2) as pbar:
                X, y = self.forecaster.prepare_data(self.featured_data)
                pbar.update(1)
                
                self.metrics = self.forecaster.train_model(X, y)
                predictions = self.forecaster.predict(X)
                pbar.update(1)
            
            # Format predictions based on dataset type
            self.predictions = self._format_predictions(predictions)
            
            # 4. Generate Insights
            with tqdm(desc="Generating insights", total=1) as pbar:
                self.insights = self.insight_generator.generate_insights(
                    self.metrics, self.predictions, self.dataset_type
                )
                pbar.update(1)
            
            # 5. Create Visualizations
            with tqdm(desc="Creating visualizations", total=1) as pbar:
                self.dashboard.save_visualizations(
                    self.featured_data, 
                    self.predictions,
                    output_dir=str(self.output_dir / "visualizations")
                )
                pbar.update(1)
            
            # 6. Save Results
            if save_artifacts:
                results.update(self._save_artifacts())
            
            # 7. Compile Results
            results.update({
                'metrics': self.metrics,
                'insights': self.insights,
                'status': 'success'
            })
            
            logger.info("Pipeline execution completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            results = {
                'status': 'failed',
                'error': str(e)
            }
            
        return results

    def _format_predictions(self, predictions) -> pd.DataFrame:
        """Format predictions based on dataset type."""
        if self.dataset_type == 'hr':
            return pd.DataFrame({
                'attrition_probability': predictions,
                'department': self.featured_data['department']
            })
        
        elif self.dataset_type == 'procurement':
            return pd.DataFrame({
                'predicted_spend': predictions,
                'vendor_id': self.featured_data['vendor_id'],
                'spend_variance': (predictions - self.featured_data['spend_amount']) / self.featured_data['spend_amount']
            })
        
        elif self.dataset_type == 'finance':
            return pd.DataFrame({
                'predicted_profit': predictions,
                'quarter': self.featured_data['quarter']
            })
        
        return pd.DataFrame(predictions)

    def _save_artifacts(self) -> Dict[str, str]:
        """Save pipeline artifacts to disk."""
        artifact_paths = {}
        
        # Save predictions
        predictions_path = self.output_dir / "forecasts.csv"
        self.predictions.to_csv(predictions_path)
        artifact_paths['predictions_file'] = str(predictions_path)
        
        # Save insights
        insights_path = self.output_dir / "insights_report.txt"
        with open(insights_path, 'w') as f:
            f.write(self.insights)
        artifact_paths['insights_file'] = str(insights_path)
        
        # Save model
        model_path = self.output_dir / "model.joblib"
        self.forecaster.save_model(str(model_path))
        artifact_paths['model_file'] = str(model_path)
        
        return artifact_paths

    def load_model(self, model_path: str):
        """Load a previously trained model."""
        self.forecaster.load_model(model_path)

    def generate_dashboard(self):
        """Generate and display the interactive dashboard."""
        if any(v is None for v in [self.featured_data, self.predictions, self.metrics]):
            raise ValueError("Pipeline must be run before generating dashboard")
            
        self.dashboard.create_dashboard(
            self.featured_data,
            self.predictions,
            self.metrics
        )

if __name__ == "__main__":
    # Example usage
    pipeline = Pipeline(dataset_type="hr")
    results = pipeline.run()
    
    print("\nPipeline Results:")
    print(f"Status: {results['status']}")
    if results['status'] == 'success':
        print(f"\nModel Metrics:")
        print(results['metrics'])
        print(f"\nGenerated Insights:")
        print(results['insights'])
        print(f"\nArtifact Locations:")
        for key, path in results.items():
            if key.endswith('_file'):
                print(f"{key}: {path}")