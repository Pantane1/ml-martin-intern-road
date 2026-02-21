# House Price Prediction - Data Preprocessing
## Level 1 - Task 1: Data Preprocessing for Machine Learning

### 📋 Task Description
Preprocess a raw dataset to make it ready for machine learning models.

### 🎯 Objectives
- ✅ Handle missing data (filling with mean/median, dropping)
- ✅ Encode categorical variables (one-hot encoding / label encoding)
- ✅ Normalize/standardize numerical features
- ✅ Split dataset into training and testing sets

### 📁 Project Structure
level-1-task-1/
├── data/
│ └── house_prediction.csv # Raw dataset
├── notebooks/
│ └── data_preprocessing.ipynb # Jupyter notebook with step-by-step analysis
├── src/
│ ├── init.py
│ └── preprocess.py # Modular preprocessing script
├── output/
│ ├── X_train.csv # Training features
│ ├── X_test.csv # Testing features
│ ├── y_train.csv # Training target
│ ├── y_test.csv # Testing target
│ ├── house_prices_processed.csv # Full processed dataset
│ └── scaler_params.csv # Scaler parameters
├── requirements.txt # Dependencies
└── README.md # Project documentation

text

### 🛠️ Tools Used
- Python 3.8+
- pandas - Data manipulation
- scikit-learn - Preprocessing and splitting
- matplotlib/seaborn - Visualization
- numpy - Numerical operations

### 📊 Dataset Information
- **Source**: Boston Housing Dataset
- **Samples**: 506 instances
- **Features**: 13 features + 1 target
- **Target**: MEDV (Median value of owner-occupied homes in $1000's)

### 📈 Preprocessing Steps
1. **Data Loading & Exploration**
   - Loaded CSV with column names
   - Checked data types and basic statistics
   - No missing values found

2. **Categorical Encoding**
   - One-hot encoding applied to CHAS and RAD columns
   - CHAS (Charles River dummy variable) - binary
   - RAD (Index of accessibility to radial highways) - ordinal

3. **Feature Scaling**
   - Standardization (Z-score normalization) applied
   - Mean = 0, Standard Deviation = 1 for all features

4. **Train-Test Split**
   - 80% training, 20% testing
   - Random state = 42 for reproducibility

### 📝 Results
- **Original Shape**: (506, 14)
- **Processed Shape**: (506, 24) after one-hot encoding
- **Training Samples**: 404
- **Testing Samples**: 102

### 🚀 How to Run
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run preprocessing script: `python src/preprocess.py`
4. Or explore step-by-step: `jupyter notebook notebooks/data_preprocessing.ipynb`

### 📌 Key Learnings
- Importance of handling missing data
- Difference between one-hot and label encoding
- Impact of feature scaling on model performance
- Train-test split importance for model evaluation

### 🔗 Links
- [GitHub Repository](https://github.com/Pantane1/ml-martin-intern-road/level1-data-preprocessing)
- [LinkedIn Post](https://www.linkedin.com/posts/pantane_denjagidev-codvedajourney-codvedaexperience-activity-7430906558240145408-Rsyh?utm_source=share&utm_medium=member_android&rcm=ACoAAFWQ7TMBh2D_Xw7HHb6_uSj4SFbyTS2SZmI)

### 📧 Contact
- **ReachMe**: [pantane](https://nf-d.netlify.app/)
- **LinkedIn**: [W-Martin](https://www.linkedin.com/in/pantane?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)

### 🏷️ Hashtags
[#denjagidev](wamuhu-martin.vercel.app/)[#CodvedaJourney](https://www.linkedin.com/posts/pantane_denjagidev-codvedajourney-codvedaexperience-activity-7430906558240145408-Rsyh?utm_source=share&utm_medium=member_android&rcm=ACoAAFWQ7TMBh2D_Xw7HHb6_uSj4SFbyTS2SZmI)  [#CodvedaExperience](https://codveda.com/) [#FutureWithCodveda](https://www.linkedin.com/company/codveda-technologies/) [#MachineLearning](https://docs.google.com/forms/d/e/1FAIpQLSeDPD-A_5tMzzWz5LG-LyvcsBm3yiRPfePARfidSdWbicK5Pg/alreadyresponded) [#Python](https://www.linkedin.com/showcase/pythoneyehunts/) [#ScikitLearn]() [#DataPreprocessing]()


Step 5: Git Commands
bash
#### Initialize git repository
-git init
-Add all files
-git add .
-Commit
-git commit -m "Initial commit: Data preprocessing for house price prediction"
-Add remote 
-git remote [add origin](https://github.com/pantane1/level-1-task-1.git)
-Push
-git branch -M main
-git push -u origin main


<p align="center">
  <a href="#"><img src="https://github.com/Pantane1/nf/blob/main/public/ph.png" alt="ph-logo">
</p>

<p align="center">
  <a href="#"><img src="http://readme-typing-svg.herokuapp.com?color=ACAF50&center=true&vCenter=true&multiline=false&lines=Built+Different" alt="pantane">
</p>
