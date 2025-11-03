# AI Business Operations Intelligence Suite 🚀

An end-to-end business intelligence solution combining data analytics, ML forecasting, and AI-powered insights using free, open datasets and Hugging Face models.

## 🎯 Overview

This project creates a comprehensive business operations intelligence tool that:

- Analyzes real-world business data (HR attrition, procurement spend, or financial operations)
- Uses ML models to forecast operational risks and trends
- Leverages Hugging Face Transformers for executive-friendly insights
- Generates interactive analytics dashboards

## 📊 Supported Datasets

The system is designed to work with the following public datasets from Kaggle:

1. **IBM HR Analytics Employee Attrition & Performance**
   - Predicts employee turnover and workforce stability
   - [Dataset Link](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

2. **Procurement Spend Analysis**
   - Forecasts spending patterns and vendor risks
   - [Dataset Link](https://www.kaggle.com/datasets/awadhi123/procurement-spend-analysis)

3. **Financial Performance**
   - Predicts contract and budget performance
   - [Dataset Link](https://www.kaggle.com/datasets/ashishraut64/financial-performance)

## 🏗️ Architecture

The system follows a modular pipeline architecture:

```
📁 Project Structure
├── data/              # Dataset storage
├── src/              # Source code
│   ├── data_loader.py        # Data ingestion
│   ├── feature_engineering.py # Feature creation
│   ├── ml_forecasting.py     # Predictive modeling
│   ├── hf_insights.py        # AI-powered insights
│   ├── dashboard.py          # Visualization
│   └── pipeline.py           # Orchestration
├── outputs/          # Generated artifacts
│   ├── forecasts.csv
│   ├── insights_report.txt
│   └── visualizations/
└── main.py          # Entry point
```

## 🛠️ Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AI-BusinessOps-Intelligence-Suite.git
   cd AI-BusinessOps-Intelligence-Suite
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to set your `LOCAL_DATA_PATH`

5. Download a dataset from Kaggle and place it in your data directory

## 🚀 Usage

Run the analysis pipeline:

```bash
python main.py --dataset hr  # or procurement, finance
```

Optional arguments:
- `--dataset`: Type of analysis ('hr', 'procurement', 'finance')
- `--output-dir`: Custom output directory
- `--load-model`: Path to previously trained model
- `--debug`: Enable debug logging

## 📈 Features

### Data Processing
- Automated data loading and preprocessing
- Missing value handling
- Feature engineering based on domain knowledge

### ML Forecasting
- HR: Attrition risk classification (RandomForest)
- Procurement: Spend forecasting (LinearRegression)
- Finance: Performance prediction (GradientBoosting)

### AI Insights
- Uses Hugging Face models for natural language insights
- Summarizes key findings and recommendations
- Supports multiple insight generation strategies

### Visualization
- Interactive Streamlit dashboards
- Plotly visualizations
- Exportable reports and charts

## 📊 Example Outputs

### HR Analytics
```
Model Accuracy: 85%
High Risk Departments: Sales (15%), Engineering (12%)
Recommended Actions: Focus on work-life balance improvements
```

### Procurement
```
Spend Forecast Accuracy: RMSE $12,345
Anomalous Vendors: 3 identified
Top Saving Opportunity: Office Supplies ($50,000)
```

### Finance
```
Budget Forecast R² Score: 0.89
Risk Areas: Q4 Marketing Budget
Trend: Positive profit growth expected
```

## 🔄 Future Improvements

- [ ] Multi-dataset integration
- [ ] Anomaly detection for fraud
- [ ] Docker containerization
- [ ] Deploy dashboard to Hugging Face Spaces
- [ ] Add more advanced ML models
- [ ] Implement real-time monitoring

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- IBM for the HR Analytics dataset
- Kaggle community for procurement and finance datasets
- Hugging Face for transformer models
- Streamlit team for the dashboard framework