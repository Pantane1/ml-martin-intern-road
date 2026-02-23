"""
Master script to run complete pipeline from raw data to trained model
"""

from src.preprocess import DataPreprocessor
from src.train_model import LinearRegressionModel
import os

def main():
    """Run complete preprocessing and modeling pipeline"""
    
    print("\n" + "="*60)
    print("🏠 HOUSE PRICE PREDICTION - COMPLETE PIPELINE")
    print("="*60)
    
    # Step 1: Preprocess raw data
    print("\n📦 STEP 1: PREPROCESSING RAW DATA")
    print("-"*40)
    
    preprocessor = DataPreprocessor('data/house_prediction_raw.csv')
    X_train, X_test, y_train, y_test = preprocessor.run_pipeline()
    
    # Step 2: Train and evaluate model
    print("\n📦 STEP 2: TRAINING LINEAR REGRESSION MODEL")
    print("-"*40)
    
    model_pipeline = LinearRegressionModel()
    model_pipeline.run_pipeline()
    
    print("\n" + "="*60)
    print("🎉🎉🎉 COMPLETE PIPELINE EXECUTED SUCCESSFULLY! 🎉🎉🎉")
    print("="*60)

if __name__ == "__main__":
    main()