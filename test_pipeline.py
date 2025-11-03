"""
Simple test script for the HR Analytics pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    """Load the HR dataset."""
    data_path = os.path.join(os.getenv('LOCAL_DATA_PATH', '/Users/michaelromero/Documents/AI_BusinessOps_Data'), 
                            'hr_attrition.csv')
    return pd.read_csv(data_path)

def prepare_features(df):
    """Prepare features for model training."""
    # Create new features
    df['tenure_ratio'] = df['YearsInCurrentRole'] / (df['TotalWorkingYears'] + 1)
    df['company_change_rate'] = df['NumCompaniesWorked'] / (df['TotalWorkingYears'] + 1)
    
    # Select features for model
    feature_cols = ['tenure_ratio', 'company_change_rate', 'MonthlyIncome', 
                   'TotalWorkingYears', 'YearsInCurrentRole']
    
    # Convert target to numeric
    target = (df['Attrition'] == 'Yes').astype(int)
    
    return df[feature_cols], target

def train_model(X, y):
    """Train a Random Forest model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    return model, metrics, X_test, y_test, y_prob

def create_visualizations(df, predictions, probabilities):
    """Create and save visualizations."""
    output_dir = 'outputs/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Department-wise Attrition
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Department', hue='Attrition')
    plt.title('Attrition by Department')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attrition_by_department.png'))
    plt.close()
    
    # 2. Monthly Income vs Years
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='TotalWorkingYears', y='MonthlyIncome', hue='Attrition')
    plt.title('Monthly Income vs Total Working Years')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'income_vs_years.png'))
    plt.close()
    
    # 3. Prediction Probability Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(probabilities, bins=30)
    plt.title('Distribution of Attrition Probabilities')
    plt.xlabel('Probability of Attrition')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attrition_probabilities.png'))
    plt.close()

def main():
    # 1. Load Data
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} records")
    
    # 2. Prepare Features
    print("\nPreparing features...")
    X, y = prepare_features(df)
    
    # 3. Train Model
    print("\nTraining model...")
    model, metrics, X_test, y_test, y_prob = train_model(X, y)
    
    # 4. Print Results
    print("\nModel Performance:")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print("\nDetailed Classification Report:")
    print(metrics['classification_report'])
    
    # 5. Create Visualizations
    print("\nCreating visualizations...")
    create_visualizations(df, y_test, y_prob)
    print("Visualizations saved to outputs/visualizations/")
    
    # 6. Generate Risk Assessment
    high_risk_threshold = 0.7
    high_risk_count = (y_prob > high_risk_threshold).sum()
    print(f"\nRisk Assessment:")
    print(f"High Risk Employees (>70% probability): {high_risk_count}")
    print(f"Percentage of High Risk: {high_risk_count/len(y_prob):.1%}")

if __name__ == "__main__":
    main()