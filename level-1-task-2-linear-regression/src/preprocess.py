"""
Data Preprocessing Module for House Price Prediction
Handles all preprocessing steps independently
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Complete data preprocessing pipeline for house price dataset
    """
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        
    def load_data(self):
        """Load the raw dataset"""
        logger.info("Loading raw data...")
        try:
            self.df = pd.read_csv(self.file_path, header=None, delim_whitespace=True)
            self.df.columns = self.column_names
            logger.info(f"✅ Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise
    
    def explore_data(self):
        """Print basic data information"""
        print("\n" + "="*60)
        print("DATA EXPLORATION")
        print("="*60)
        print(f"\n📊 Dataset Shape: {self.df.shape}")
        print(f"\n📋 Column Names:")
        for col in self.df.columns:
            print(f"   • {col}")
        
        print(f"\n📈 Basic Statistics:")
        print(self.df.describe().round(2))
        
        print(f"\n🔍 Data Types:")
        print(self.df.dtypes)
        
        print(f"\n🔎 Missing Values:")
        print(self.df.isnull().sum())
    
    def handle_missing_values(self, strategy='mean'):
        """Handle any missing values (though none in this dataset)"""
        logger.info(f"Checking for missing values...")
        missing = self.df.isnull().sum().sum()
        if missing == 0:
            logger.info("✅ No missing values found")
            return self.df
        
        logger.info(f"Found {missing} missing values. Handling with {strategy}...")
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'mean':
            self.df = self.df.fillna(self.df.mean())
        elif strategy == 'median':
            self.df = self.df.fillna(self.df.median())
        
        logger.info(f"Shape after handling missing values: {self.df.shape}")
        return self.df
    
    def encode_categorical(self):
        """Encode categorical variables (CHAS and RAD)"""
        logger.info("Encoding categorical variables...")
        
        # CHAS is already binary (0/1) - no encoding needed
        # RAD is ordinal - we'll keep as is, but note it's categorical
        
        print("\n🏷️ Categorical Variables Info:")
        print(f"   CHAS (Charles River): {self.df['CHAS'].unique()} values")
        print(f"   RAD (Highway Access): {sorted(self.df['RAD'].unique())}")
        
        # For demonstration, we'll create dummy variables for RAD
        logger.info("Applying one-hot encoding to RAD...")
        rad_dummies = pd.get_dummies(self.df['RAD'], prefix='RAD', drop_first=False)
        self.df = pd.concat([self.df.drop('RAD', axis=1), rad_dummies], axis=1)
        
        logger.info(f"Shape after encoding: {self.df.shape}")
        return self.df
    
    def detect_outliers(self, threshold=3):
        """Detect outliers using z-score method"""
        logger.info("Detecting outliers...")
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_summary = {}
        
        for col in numerical_cols:
            z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
            outliers = z_scores > threshold
            outlier_summary[col] = {
                'count': outliers.sum(),
                'percentage': (outliers.sum() / len(self.df)) * 100
            }
        
        print("\n📊 Outlier Summary (Z-score > 3):")
        print("-" * 50)
        for col, stats in outlier_summary.items():
            if stats['count'] > 0:
                print(f"   {col:12}: {stats['count']:3} outliers ({stats['percentage']:.1f}%)")
        
        return outlier_summary
    
    def scale_features(self):
        """Scale features using StandardScaler"""
        logger.info("Scaling features using StandardScaler...")
        
        # Separate features and target
        X = self.df.drop('MEDV', axis=1)
        y = self.df['MEDV']
        
        # Fit and transform
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Recombine with target
        self.df = pd.concat([X_scaled, y], axis=1)
        
        logger.info("✅ Features scaled (mean=0, std=1)")
        return X_scaled, y
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        logger.info(f"Splitting data (test_size={test_size})...")
        
        X = self.df.drop('MEDV', axis=1)
        y = self.df['MEDV']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"✅ Training set: {self.X_train.shape}")
        logger.info(f"✅ Testing set: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_processed_data(self, output_dir='output/processed'):
        """Save all processed datasets"""
        os.makedirs(output_dir, exist_ok=True)
        
        self.X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
        self.X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
        self.y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
        self.y_test.to_csv(f'{output_dir}/y_test.csv', index=False)
        
        # Save scaler parameters
        scaler_params = pd.DataFrame({
            'feature': self.X_train.columns,
            'mean': self.scaler.mean_,
            'scale': self.scaler.scale_
        })
        scaler_params.to_csv(f'{output_dir}/scaler_params.csv', index=False)
        
        logger.info(f"✅ All processed data saved to {output_dir}")
    
    def run_pipeline(self):
        """Run the complete preprocessing pipeline"""
        print("\n" + "="*60)
        print("🏠 HOUSE PRICE PREDICTION - PREPROCESSING PIPELINE")
        print("="*60)
        
        self.load_data()
        self.explore_data()
        self.handle_missing_values()
        self.encode_categorical()
        self.detect_outliers()
        self.scale_features()
        self.split_data()
        self.save_processed_data()
        
        print("\n" + "="*60)
        print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return self.X_train, self.X_test, self.y_train, self.y_test


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor('data/house_prediction_raw.csv')
    X_train, X_test, y_train, y_test = preprocessor.run_pipeline()