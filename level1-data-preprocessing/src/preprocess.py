"""
House Price Prediction - Data Preprocessing Module
Task 1: Data Preprocessing for Machine Learning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    A class to handle all data preprocessing tasks for machine learning
    """
    
    def __init__(self, file_path):
        """
        Initialize the preprocessor with data file path
        
        Parameters:
        file_path (str): Path to the CSV file
        """
        self.file_path = file_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                             'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        
    def load_data(self):
        """Load the dataset"""
        try:
            self.df = pd.read_csv(self.file_path, header=None, delim_whitespace=True)
            self.df.columns = self.column_names
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def explore_data(self):
        """Explore basic statistics of the dataset"""
        logger.info("Exploring data...")
        print("\n" + "="*50)
        print("DATASET OVERVIEW")
        print("="*50)
        print(f"Shape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nData Types:\n{self.df.dtypes}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()}")
        print(f"\nBasic Statistics:\n{self.df.describe()}")
        
    def handle_missing_data(self, strategy='mean'):
        """
        Handle missing values in the dataset
        
        Parameters:
        strategy (str): 'mean', 'median', 'most_frequent', or 'drop'
        """
        logger.info(f"Handling missing data using strategy: {strategy}")
        
        if strategy == 'drop':
            initial_shape = self.df.shape
            self.df = self.df.dropna()
            logger.info(f"Dropped rows with missing values. Shape changed from {initial_shape} to {self.df.shape}")
        else:
            # Use SimpleImputer for other strategies
            imputer = SimpleImputer(strategy=strategy)
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numerical_cols] = imputer.fit_transform(self.df[numerical_cols])
            logger.info(f"Missing values filled using {strategy} strategy")
        
        return self.df
    
    def encode_categorical(self, method='onehot', columns=None):
        """
        Encode categorical variables
        
        Parameters:
        method (str): 'onehot' or 'label'
        columns (list): List of column names to encode
        """
        if columns is None:
            columns = ['CHAS', 'RAD']  # Known categorical columns
        
        logger.info(f"Encoding categorical columns: {columns} using {method} method")
        
        if method == 'onehot':
            self.df = pd.get_dummies(self.df, columns=columns, prefix=columns)
        elif method == 'label':
            label_encoders = {}
            for col in columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                label_encoders[col] = le
            logger.info(f"Label encoders created for {columns}")
        else:
            raise ValueError("Method must be 'onehot' or 'label'")
        
        logger.info(f"Shape after encoding: {self.df.shape}")
        return self.df
    
    def detect_outliers(self, threshold=3):
        """
        Detect outliers using z-score method
        
        Parameters:
        threshold (float): Z-score threshold for outlier detection
        """
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
        
        print("\n" + "="*50)
        print("OUTLIER DETECTION SUMMARY")
        print("="*50)
        for col, stats in outlier_summary.items():
            print(f"{col}: {stats['count']} outliers ({stats['percentage']:.2f}%)")
        
        return outlier_summary
    
    def scale_features(self, method='standard'):
        """
        Scale numerical features
        
        Parameters:
        method (str): 'standard' for StandardScaler, 'minmax' for MinMaxScaler
        """
        logger.info(f"Scaling features using {method} method")
        
        # Separate features and target
        X = self.df.drop('MEDV', axis=1)
        y = self.df['MEDV']
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Recombine with target
        self.df = pd.concat([X_scaled, y], axis=1)
        
        logger.info(f"Features scaled. Shape: {self.df.shape}")
        logger.info(f"Scaler statistics saved")
        
        return X_scaled, y
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Parameters:
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        """
        logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
        
        X = self.df.drop('MEDV', axis=1)
        y = self.df['MEDV']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        logger.info(f"Training set size: {self.X_train.shape}")
        logger.info(f"Testing set size: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_processed_data(self, output_dir='../output'):
        """
        Save all processed datasets
        
        Parameters:
        output_dir (str): Directory to save processed data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save splits
        self.X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
        self.X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
        self.y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
        self.y_test.to_csv(f'{output_dir}/y_test.csv', index=False)
        
        # Save full processed dataset
        self.df.to_csv(f'{output_dir}/house_prices_processed.csv', index=False)
        
        # Save scaler parameters
        if self.scaler:
            scaler_params = {
                'mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
                'scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
                'min': self.scaler.min_.tolist() if hasattr(self.scaler, 'min_') else None,
                'max': self.scaler.max_.tolist() if hasattr(self.scaler, 'max_') else None
            }
            
            pd.DataFrame([scaler_params]).to_csv(f'{output_dir}/scaler_params.csv')
        
        logger.info(f"All processed data saved to {output_dir}")
    
    def generate_report(self):
        """Generate a comprehensive preprocessing report"""
        report = {
            'original_shape': self.df.shape if self.df is not None else None,
            'columns': list(self.df.columns) if self.df is not None else None,
            'missing_values_handled': self.df.isnull().sum().sum() == 0 if self.df is not None else False,
            'categorical_encoded': any('CHAS' in col or 'RAD' in col for col in self.df.columns if self.df is not None),
            'scaling_applied': self.scaler is not None,
            'train_size': self.X_train.shape if self.X_train is not None else None,
            'test_size': self.X_test.shape if self.X_test is not None else None
        }
        
        print("\n" + "="*50)
        print("FINAL PREPROCESSING REPORT")
        print("="*50)
        for key, value in report.items():
            print(f"{key}: {value}")
        
        return report

def main():
    """
    Main function to execute all preprocessing steps
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor('../data/house_prediction.csv')
    
    try:
        # Load data
        preprocessor.load_data()
        
        # Explore data
        preprocessor.explore_data()
        
        # Handle missing data
        preprocessor.handle_missing_data(strategy='mean')
        
        # Encode categorical variables
        preprocessor.encode_categorical(method='onehot', columns=['CHAS', 'RAD'])
        
        # Detect outliers
        preprocessor.detect_outliers(threshold=3)
        
        # Scale features
        preprocessor.scale_features(method='standard')
        
        # Split data
        preprocessor.split_data(test_size=0.2, random_state=42)
        
        # Save processed data
        preprocessor.save_processed_data()
        
        # Generate report
        preprocessor.generate_report()
        
        logger.info("Data preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {e}")
        raise

if __name__ == "__main__":
    main()