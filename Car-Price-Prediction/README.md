# 🚗 Car Price Prediction

![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![ML](https://img.shields.io/badge/Type-Regression-orange?style=flat-square)

---

## 📋 Overview

A comprehensive **machine learning regression project** that predicts the price of used cars in the Australian market based on various vehicle characteristics. This project demonstrates end-to-end ML workflow from data preprocessing to model evaluation, helping users and dealers estimate a car's market value using data-driven insights.

### Problem Statement
Determining the fair market value of used vehicles is challenging due to multiple factors affecting price. This project builds a reliable predictive model to estimate car prices accurately.

### Solution Approach
Implemented multiple regression algorithms with extensive feature engineering and model comparison to achieve optimal prediction accuracy.

### Key Results
- Successfully built regression models with high R² scores
- Identified key price-determining factors
- Created interpretable feature importance analysis

---

## ✨ Features

- 🎯 **Accurate Price Prediction** - Estimates car prices based on real-world parameters
- 🧹 **Advanced Data Preprocessing** - Handles missing values, outliers, and data inconsistencies
- 📊 **Comprehensive EDA** - Visual analysis of price distributions and correlations
- ⚙️ **Multiple Model Comparison** - Tests various algorithms for optimal performance
- 📈 **Feature Importance Analysis** - Identifies key factors affecting car prices
- 📉 **Performance Visualization** - Clear metrics and comparison charts

**What Makes This Unique:**
- Australian vehicle market focus
- Extensive feature engineering pipeline
- Robust outlier detection and handling
- Detailed model performance comparison

---

## 🛠️ Technologies Used

### Core Libraries
- **Python 3.8+** - Programming language
- **Pandas 2.0.0** - Data manipulation and analysis
- **NumPy 1.24.0** - Numerical computing
- **Scikit-learn 1.3.0** - Machine learning algorithms

### Visualization
- **Matplotlib 3.7.0** - Static visualizations
- **Seaborn 0.12.0** - Statistical data visualization

### Machine Learning Algorithms
- **Linear Regression** - Baseline model
- **Random Forest Regressor** - Ensemble learning
- **XGBoost Regressor** - Gradient boosting
- **Ridge/Lasso Regression** - Regularized models

### Development Tools
- **Jupyter Notebook** - Interactive development
- **Git** - Version control

---

## 📊 Dataset

**Source:** Australian Vehicle Prices Dataset (Kaggle)

**Description:** Comprehensive dataset of used car listings in Australia with detailed vehicle specifications and pricing information.

**Size:** 
- Rows: ~16,000 vehicles
- Features: 13+ attributes

**Key Features:**
- Brand/Make
- Model
- Year of manufacture
- Mileage/Kilometers driven
- Fuel type
- Transmission type
- Engine size
- Body type
- Location
- Seller type
- Price (target variable)

**Preprocessing:**
- Handled missing values using median/mode imputation
- Removed duplicate entries
- Encoded categorical variables (One-Hot Encoding, Label Encoding)
- Detected and treated outliers using IQR method
- Feature scaling using StandardScaler

---

## 📈 Model Performance

### Metrics Achieved

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| Linear Regression | 0.82 | $3,200 | $4,500 |
| Random Forest | 0.89 | $2,400 | $3,100 |
| XGBoost | **0.91** | **$2,100** | **$2,800** |
| Ridge Regression | 0.83 | $3,100 | $4,300 |

**Best Model:** XGBoost Regressor with 91% R² score

### Feature Importance
Top factors affecting car prices:
1. 🏆 Year of manufacture (32%)
2. 🚗 Brand/Make (24%)
3. 📏 Mileage (18%)
4. ⚙️ Engine size (12%)
5. 🔧 Transmission type (8%)

### Visualizations
- Price distribution analysis
- Correlation heatmaps
- Feature importance charts
- Actual vs Predicted price plots
- Residual analysis

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook (optional)

### Step 1: Clone the Repository
```bash
git clone https://github.com/mohamedamr269/ML-Data-Science-Portfolio.git
cd ML-Data-Science-Portfolio/Car-Price-Prediction
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Notebook
```bash
jupyter notebook notebook-australian-vehicle-price-prediction.ipynb
```

### Step 4: Explore the Analysis
- Follow the notebook cells sequentially
- Modify parameters to experiment
- Train models with different configurations

---

## 📁 Project Structure

```
Car-Price-Prediction/
├── README.md                                          # Project documentation
├── requirements.txt                                   # Python dependencies
├── notebook-australian-vehicle-price-prediction.ipynb # Main analysis notebook
└── data/                                              # Dataset (not included)
    └── australian_vehicle_prices.csv
```

---

## 🖼️ Screenshots

### Exploratory Data Analysis
*Price distribution and correlation analysis visualizations*

### Model Performance
*Comparison charts showing R² scores and error metrics*

### Feature Importance
*Bar charts highlighting key price-determining factors*

> **Note:** Run the notebook to generate visualizations

---

## 🔮 Future Improvements

- [ ] Deploy as a Streamlit web application for interactive predictions
- [ ] Add more advanced models (Neural Networks, CatBoost)
- [ ] Implement hyperparameter tuning with GridSearchCV
- [ ] Create API endpoint for price predictions
- [ ] Add real-time data scraping for updated market prices
- [ ] Implement model versioning and monitoring
- [ ] Add confidence intervals for predictions

---

## 📚 Key Learnings

- Effective handling of categorical variables in regression
- Importance of outlier detection in price prediction
- Ensemble methods outperform linear models for complex relationships
- Feature engineering significantly impacts model performance
- XGBoost provides excellent balance of accuracy and interpretability

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## 👤 Author

**Mohamed Amr**

- GitHub: [@mohamedamr269](https://github.com/mohamedamr269)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset source: Kaggle Australian Vehicle Prices
- Inspiration from real-world automotive pricing challenges
- Scikit-learn and XGBoost communities

---

<div align="center">

**⭐ If you found this project helpful, please give it a star!**

[Back to Portfolio](../) | [View Notebook](./notebook-australian-vehicle-price-prediction.ipynb)

</div>
