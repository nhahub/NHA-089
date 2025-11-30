# CardioGuard AI - Heart Attack Risk Prediction System

A machine learning-powered clinical risk assessment system for predicting heart attack risk using patient health metrics and lifestyle factors.

![CardioGuard AI Dashboard](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![MLflow](https://img.shields.io/badge/MLflow-Enabled-orange)

## 🎯 Project Overview

CardioGuard AI is a comprehensive healthcare predictive analytics project that combines machine learning, MLOps best practices, and an interactive web dashboard to provide real-time heart attack risk assessments. The system uses a Random Forest classifier trained on Indonesian heart attack prediction data to analyze 27 different patient features.

### Key Features

- **Real-time Risk Assessment**: Interactive dashboard for instant patient risk evaluation
- **Explainable AI**: Feature importance visualization to understand prediction drivers
- **MLOps Integration**: Experiment tracking and model versioning with MLflow
- **Production-Ready**: Robust preprocessing pipeline and error handling
- **User-Friendly Interface**: Clean, professional Dash-based web application

## 📊 Model Performance

- **Algorithm**: Random Forest Classifier
- **Features**: 27 patient health metrics (vitals, lifestyle, medical history)
- **Tracking**: MLflow for experiment management and reproducibility

## 🏗️ Project Structure

```
Heart attack prediction/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── Dashboard.py                 # Main Dash application
├── train_with_mlflow.py        # Model training with MLflow tracking
├── preprocessing.py            # Data preprocessing pipeline
│
├── data/
│   └── heart_attack_prediction_indonesia.csv
│
├── models/
│   └── full_model_pipeline.joblib  # Trained model pipeline
│
├── src/                        # Dashboard source code
│   ├── callbacks.py           # Dashboard callback logic
│   ├── components.py          # UI component generators
│   ├── constants.py           # Feature definitions and styling
│   ├── data_manager.py        # Data and model loading
│   └── layout.py              # Dashboard layout
│
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── 01_EDA.ipynb           # Exploratory Data Analysis
│   ├── 02_model_development.ipynb  # Model development
│   └── 03_research.ipynb      # Research and experimentation
│
├── assets/                     # Visualization and static files
│   ├── heart_attack_correlation_heatmap.png
│   └── target_distribution_plot.png
│
└── mlruns/                     # MLflow experiment tracking
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Heart attack prediction"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (optional - pre-trained model included)
   ```bash
   python train_with_mlflow.py
   ```

4. **Run the dashboard**
   ```bash
   python Dashboard.py
   ```

5. **Access the application**
   - Open your browser to: `http://127.0.0.1:8050/`

## 📋 Usage

### Dashboard Interface

The CardioGuard AI dashboard is organized into four main sections:

1. **Patient Profile**: Age, gender, region, income level
2. **Vitals & Labs**: Blood pressure, cholesterol, glucose, EKG results
3. **Lifestyle**: Smoking, physical activity, diet, sleep, stress
4. **Medical History**: Hypertension, diabetes, family history

### Making a Prediction

1. Fill in patient information across all sections
2. Click the **"ANALYZE PATIENT"** button
3. View the risk assessment:
   - **Risk Level**: HIGH RISK or LOW RISK
   - **Probability**: Percentage likelihood of heart attack
   - **Recommendation**: Clinical guidance based on result

### Feature Importance

The right panel displays the top 10 most influential features in the model's decision-making process, helping clinicians understand which factors contribute most to the risk assessment.

## 🔬 MLOps Pipeline

### Experiment Tracking

```bash
# View MLflow UI
mlflow ui
```

Access the MLflow dashboard at `http://127.0.0.1:5000` to:
- Compare model experiments
- View metrics (accuracy, precision, recall, F1-score)
- Track model versions and artifacts

### Model Training

The training pipeline includes:
- Data preprocessing (StandardScaler for numeric, OrdinalEncoder for categorical)
- Train/test split (80/20)
- Model training with hyperparameter logging
- Automatic metric calculation and logging
- Pipeline serialization for deployment

## 🛠️ Technical Details

### Data Preprocessing

- **Numeric Features**: Standardized using `StandardScaler`
- **Categorical Features**: Encoded using `OrdinalEncoder`
- **Missing Values**: Handled as NaN for appropriate features (e.g., alcohol consumption)

### Model Pipeline

```python
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OrdinalEncoder(), CATEGORICAL_FEATURES)
    ])),
    ('classifier', RandomForestClassifier(n_estimators=50, max_depth=10))
])
```

## 📦 Deployment

### Plotly Cloud (Optional)

The project includes `dash[cloud]` extension for easy deployment to Plotly Cloud:

```bash
# Deploy to Plotly Cloud
dash-cloud deploy
```

### Local Production

For local production deployment, consider using:
- **Gunicorn**: WSGI server for production
- **Docker**: Containerization for consistent deployment
- **Nginx**: Reverse proxy for load balancing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- Mohamed Amr - Initial work

## 🙏 Acknowledgments

- Dataset: Indonesian Heart Attack Prediction Dataset
- MLflow for experiment tracking capabilities
- Plotly Dash for the interactive dashboard framework

## 📞 Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: This system is designed for educational and research purposes. Always consult with qualified healthcare professionals for medical decisions.
