import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of employees
n_employees = 1000

# Generate more realistic employee data
def generate_hr_data():
    # Employee IDs
    data = {
        'EmployeeNumber': range(1, n_employees + 1),
        
        # Departments with realistic distribution
        'Department': np.random.choice(
            ['Sales', 'Engineering', 'HR', 'Finance', 'Marketing', 'Operations'],
            size=n_employees,
            p=[0.3, 0.25, 0.1, 0.15, 0.1, 0.1]
        ),
        
        # Total Working Years (log-normal distribution for right skew)
        'TotalWorkingYears': np.random.lognormal(2, 0.7, n_employees).astype(int).clip(0, 40),
        
        # Monthly Income (based on years of experience and department)
        'MonthlyIncome': None,  # Will be set based on experience
        
        # Number of companies worked (depends on total years)
        'NumCompaniesWorked': None,  # Will be set based on years
        
        # Years in current role (must be less than total years)
        'YearsInCurrentRole': None,  # Will be set based on total years
        
        # Overtime (more likely in certain departments)
        'OverTime': None  # Will be set based on department
    }
    
    df = pd.DataFrame(data)
    
    # Set department-based salary multipliers
    dept_multipliers = {
        'Sales': 1.1,
        'Engineering': 1.3,
        'HR': 0.9,
        'Finance': 1.2,
        'Marketing': 1.0,
        'Operations': 0.95
    }
    
    # Calculate realistic salaries based on experience and department
    base_salary = 3500
    experience_factor = 800
    
    df['MonthlyIncome'] = (
        base_salary + 
        df['TotalWorkingYears'] * experience_factor * 
        df['Department'].map(dept_multipliers) * 
        np.random.normal(1, 0.1, n_employees)  # Add some random variation
    ).astype(int)
    
    # Calculate companies worked based on years of experience
    df['NumCompaniesWorked'] = (
        df['TotalWorkingYears'] / 5 * np.random.normal(1, 0.3, n_employees)
    ).clip(0, 8).astype(int)
    
    # Set years in current role (always less than total years)
    df['YearsInCurrentRole'] = (
        df['TotalWorkingYears'] * np.random.uniform(0.2, 0.8, n_employees)
    ).astype(int)
    
    # Set overtime based on department and random chance
    overtime_prob = {
        'Sales': 0.4,
        'Engineering': 0.5,
        'HR': 0.2,
        'Finance': 0.3,
        'Marketing': 0.3,
        'Operations': 0.45
    }
    
    df['OverTime'] = df.apply(
        lambda row: 'Yes' if np.random.random() < overtime_prob[row['Department']] else 'No',
        axis=1
    )
    
    # Calculate attrition risk factors
    attrition_prob = (
        # Base probability
        0.15 +
        # Higher probability if low salary for experience
        0.2 * (df['MonthlyIncome'] < df['TotalWorkingYears'] * experience_factor).astype(int) +
        # Higher probability if working overtime
        0.15 * (df['OverTime'] == 'Yes').astype(int) +
        # Higher probability if many previous companies
        0.1 * (df['NumCompaniesWorked'] > 3).astype(int) +
        # Lower probability if long tenure in current role
        -0.1 * (df['YearsInCurrentRole'] > 5).astype(int)
    )
    
    # Normalize probabilities to 0-1 range
    attrition_prob = (attrition_prob - attrition_prob.min()) / (attrition_prob.max() - attrition_prob.min())
    
    # Set attrition based on calculated probabilities
    df['Attrition'] = attrition_prob.apply(
        lambda x: 'Yes' if np.random.random() < x else 'No'
    )
    
    return df

# Generate the dataset
df = generate_hr_data()

# Add some data validation
df['YearsInCurrentRole'] = df.apply(
    lambda x: min(x['YearsInCurrentRole'], x['TotalWorkingYears']),
    axis=1
)

# Save to CSV
output_path = 'AI_BusinessOps_Data/hr_attrition.csv'
df.to_csv(output_path, index=False)
print(f"Sample dataset created at: {output_path}")
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())