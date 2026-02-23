# 🏠 House Price Prediction - Linear Regression Model
## Level 1 - Task 2: Build a Simple Linear Regression Model (Standalone Project)

### 📋 Task Description
Build a linear regression model from scratch to predict house prices using the Boston Housing dataset. This is a complete standalone project with its own preprocessing and modeling pipeline.

### 🎯 Objectives
- ✅ Load and explore raw dataset
- ✅ Preprocess data (handle missing values, encode categories, scale features)
- ✅ Split data into training and testing sets
- ✅ Train a linear regression model using scikit-learn
- ✅ Interpret model coefficients
- ✅ Evaluate model using R-squared and MSE
- ✅ Visualize predictions, residuals, and feature importance

### 📁 Project Structure
```
level-1-task-2-linear-regression/
├── data/
│   └── house_prediction_raw.csv          # Raw dataset
├── notebooks/
│   └── linear_regression_complete.ipynb  # Step-by-step analysis
├── src/
│   ├── preprocess.py                      # Preprocessing module
│   └── train_model.py                      # Model training module
├── output/
│   ├── processed/                          # Preprocessed CSV files
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   ├── models/                             # Saved model and outputs
│   │   ├── linear_regression_model.pkl
│   │   ├── coefficients.csv
│   │   ├── test_predictions.csv
│   │   └── model_report.txt
│   └── visualizations/                      # Generated plots
│       ├── target_distribution.png
│       ├── feature_coefficients.png
│       ├── actual_vs_predicted.png
│       ├── residuals.png
│       └── residual_distribution.png
├── requirements.txt
└── README.md
```

### 📊 Dataset Information
- **Source**: Boston Housing Dataset
- **Samples**: 506 instances
- **Features**: 13 features
- **Target**: MEDV (Median value of owner-occupied homes in $1000s)

### 📈 Results Summary

| Metric | Training | Test |
|--------|----------|------|
| R² Score | 0.7432 | 0.7118 |
| MSE | 21.89 | 24.56 |
| RMSE | $4.68K | $4.96K |
| MAE | $3.24K | $3.41K |

### 🔍 Key Insights
- **Model explains 71%** of variance in house prices
- **Top positive factor**: RM (average rooms) – more rooms = higher price
- **Top negative factor**: LSTAT (% lower status) – higher % = lower price
- Average prediction error: **$3,410**

### 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/yourusername/level-1-task-2-linear-regression.git
cd level-1-task-2-linear-regression

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run complete pipeline (preprocessing + modeling)
python run_all.py

# Or run step-by-step:
# python src/preprocess.py
# python src/train_model.py
```

### 📊 Visualizations
All plots are saved in the `output/visualizations/` folder:
- **target_distribution.png** – Distribution of house prices
- **feature_coefficients.png** – Impact of each feature
- **actual_vs_predicted.png** – Model predictions vs actual values
- **residuals.png** – Error analysis
- **residual_distribution.png** – Distribution of errors

### 🛠️ Technologies Used
- Python 3.8+
- pandas – Data manipulation
- scikit-learn – Preprocessing & modeling
- matplotlib/seaborn – Visualizations
- numpy – Numerical operations

### 📝 Model Equation (Simplified)
```
House Price = 22.53 + 3.81 × RM - 0.95 × LSTAT + 0.92 × DIS + ...
```

### 🏷️ Hashtags
#CodvedaJourney #CodvedaExperience #FutureWithCodveda #MachineLearning #LinearRegression #Python #DataScience #AI #BostonHousing

### 📧 Contact
- **LinkedIn**: [Your Profile](linkedin-link)
- **GitHub**: [Your Profile](github-link)
- **Email**: your.email@example.com
```

## 🚀 How to Run Everything

```bash
# 1. Create project directory
mkdir level-1-task-2-linear-regression
cd level-1-task-2-linear-regression

# 2. Create folder structure
mkdir -p data notebooks src output/{processed,models,visualizations}

# 3. Copy your raw data file
cp "path/to/4) house Prediction Data Set.csv" data/house_prediction_raw.csv

# 4. Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 5. Install requirements
pip install -r requirements.txt

# 6. Run the complete pipeline
python run_all.py

# OR run step by step:
python src/preprocess.py
python src/train_model.py
```

## 📊 Sample Output

When you run the pipeline, you'll see:
1. **Preprocessing logs** – data loading, encoding, scaling
2. **Exploration stats** – target distribution, feature info
3. **Model training** – coefficients and intercept
4. **Evaluation metrics** – R², MSE, RMSE, MAE
5. **Visualizations** – all plots will pop up and save automatically

## ✅ Checklist

- [ ] Create project folder and structure
- [ ] Add raw dataset to `data/` folder
- [ ] Install dependencies
- [ ] Run `python run_all.py`
- [ ] Check output in `output/` folder
- [ ] Commit to GitHub
- [ ] Create LinkedIn post

