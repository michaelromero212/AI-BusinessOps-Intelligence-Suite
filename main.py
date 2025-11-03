"""
Main entry point for the AI Business Operations Intelligence Suite.
"""

import argparse
import logging
from pathlib import Path
import sys
from dotenv import load_dotenv
import os
from src.pipeline import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up the environment and verify requirements."""
    # Load environment variables
    load_dotenv()
    
    # Check for required environment variables
    required_vars = ['LOCAL_DATA_PATH']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please check your .env file")
        sys.exit(1)
    
    # Create required directories
    data_path = Path(os.getenv('LOCAL_DATA_PATH'))
    data_path.mkdir(parents=True, exist_ok=True)
    
    outputs_path = Path("outputs")
    outputs_path.mkdir(parents=True, exist_ok=True)
    (outputs_path / "visualizations").mkdir(exist_ok=True)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Business Operations Intelligence Suite"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['hr', 'procurement', 'finance'],
        default='hr',
        help='Type of dataset to analyze'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Directory to save outputs'
    )
    
    parser.add_argument(
        '--load-model',
        type=str,
        help='Path to a previously trained model to load'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    return parser.parse_args()

def check_dataset_availability(dataset_type: str):
    """Check if the required dataset is available."""
    data_path = Path(os.getenv('LOCAL_DATA_PATH'))
    dataset_files = {
        'hr': 'hr_attrition.csv',
        'procurement': 'procurement.csv',
        'finance': 'finance.csv'
    }
    
    dataset_file = data_path / dataset_files[dataset_type]
    
    if not dataset_file.exists():
        logger.error(f"Dataset not found: {dataset_file}")
        print("\nTo use this dataset, please:")
        print(f"1. Download the dataset from Kaggle")
        print(f"2. Place it in {data_path} as {dataset_files[dataset_type]}")
        print("\nKaggle dataset links:")
        print("HR Attrition: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset")
        print("Procurement: https://www.kaggle.com/datasets/awadhi123/procurement-spend-analysis")
        print("Financial Performance: https://www.kaggle.com/datasets/ashishraut64/financial-performance")
        sys.exit(1)

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up environment
    setup_environment()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    logger.info(f"Starting analysis with {args.dataset} dataset")
    
    # Check dataset availability
    check_dataset_availability(args.dataset)
    
    try:
        # Initialize pipeline
        pipeline = Pipeline(
            dataset_type=args.dataset,
            output_dir=args.output_dir
        )
        
        # Load previous model if specified
        if args.load_model:
            logger.info(f"Loading model from {args.load_model}")
            pipeline.load_model(args.load_model)
        
        # Run pipeline
        results = pipeline.run()
        
        if results['status'] == 'success':
            logger.info("Analysis completed successfully!")
            print("\nResults Summary:")
            print("-" * 50)
            
            # Print metrics
            print("\nModel Performance:")
            for metric, value in results['metrics'].items():
                if isinstance(value, float):
                    print(f"{metric}: {value:.4f}")
            
            # Print insights
            print("\nKey Insights:")
            print(results['insights'])
            
            # Print artifact locations
            print("\nOutput Files:")
            for key, path in results.items():
                if key.endswith('_file'):
                    print(f"{key}: {path}")
            
            # Generate dashboard
            logger.info("Launching dashboard...")
            pipeline.generate_dashboard()
            
        else:
            logger.error(f"Pipeline failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        if args.debug:
            raise
        sys.exit(1)

if __name__ == "__main__":
    main()