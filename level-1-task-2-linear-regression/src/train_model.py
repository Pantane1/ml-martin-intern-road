"""
Linear Regression Model Training Module
Trains and evaluates model on preprocessed data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LinearRegressionModel:
    """
    Complete linear regression pipeline for house price prediction
    """
    
    def __init__(self, data_dir='output/processed', model_dir='output/models', viz_dir='output/visualizations'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.viz_dir = viz_dir
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
    
    def load_data(self):
        """Load preprocessed data"""
        logger.info("Loading preprocessed data...")
        
        self.X_train = pd.read_csv(f'{self.data_dir}/X_train.csv')
        self.X_test = pd.read_csv(f'{self.data_dir}/X_test.csv')
        self.y_train = pd.read_csv(f'{self.data_dir}/y_train.csv').squeeze()
        self.y_test = pd.read_csv(f'{self.data_dir}/y_test.csv').squeeze()
        
        logger.info(f"✅ Training features: {self.X_train.shape}")
        logger.info(f"✅ Testing features: {self.X_test.shape}")
        logger.info(f"✅ Training target: {self.y_train.shape}")
        logger.info(f"✅ Testing target: {self.y_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def explore_target(self):
        """Explore target variable distribution"""
        print("\n" + "="*60)
        print("TARGET VARIABLE EXPLORATION")
        print("="*60)
        
        print(f"\n📊 Training Set - MEDV (house prices in $1000s):")
        print(self.y_train.describe())
        
        # Visualize
        plt.figure(figsize=(10, 6))
        sns.histplot(self.y_train, kde=True, color='skyblue')
        plt.axvline(self.y_train.mean(), color='red', linestyle='--', 
                   label=f"Mean: ${self.y_train.mean():.2f}K")
        plt.axvline(self.y_train.median(), color='green', linestyle='--', 
                   label=f"Median: ${self.y_train.median():.2f}K")
        plt.title('Distribution of House Prices (Training Set)', fontsize=14, fontweight='bold')
        plt.xlabel('Median House Price ($1000s)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(f'{self.viz_dir}/target_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"📊 Target distribution plot saved")
    
    def train_model(self):
        """Train linear regression model"""
        logger.info("Training Linear Regression model...")
        
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        
        print("\n" + "="*60)
        print("MODEL TRAINING COMPLETE")
        print("="*60)
        print(f"\n📊 Model Parameters:")
        print(f"   • Intercept (bias): ${self.model.intercept_:.4f}K")
        print(f"   • Number of features: {len(self.model.coef_)}")
        
        return self.model
    
    def analyze_coefficients(self):
        """Analyze and visualize feature coefficients"""
        print("\n" + "="*60)
        print("FEATURE COEFFICIENTS ANALYSIS")
        print("="*60)
        
        # Create coefficients dataframe
        coef_df = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Coefficient': self.model.coef_
        })
        coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
        coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
        
        print("\n📈 Top 10 Most Influential Features:")
        print("-" * 60)
        for idx, row in coef_df.head(10).iterrows():
            direction = "🔼 increases" if row['Coefficient'] > 0 else "🔽 decreases"
            print(f"   {row['Feature']:15}: {row['Coefficient']:8.4f}  {direction}")
        
        print("\n📌 Interpretation:")
        print("   • Positive coefficient → As feature increases, house price increases")
        print("   • Negative coefficient → As feature increases, house price decreases")
        
        # Visualize coefficients
        plt.figure(figsize=(12, 8))
        top_features = coef_df.head(15)
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_features['Coefficient']]
        
        plt.barh(range(len(top_features)), top_features['Coefficient'], color=colors)
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Coefficient Value (Impact on House Price in $1000s)', fontsize=12)
        plt.title('Top 15 Feature Coefficients - Impact on House Price', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(top_features['Coefficient']):
            plt.text(v + (0.5 if v > 0 else -2.5), i, f'{v:.2f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.viz_dir}/feature_coefficients.png', dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"📊 Coefficients plot saved")
        
        return coef_df
    
    def make_predictions(self):
        """Make predictions on train and test sets"""
        logger.info("Making predictions...")
        
        self.y_train_pred = self.model.predict(self.X_train)
        self.y_test_pred = self.model.predict(self.X_test)
        
        print("\n" + "="*60)
        print("SAMPLE PREDICTIONS")
        print("="*60)
        
        # Show random samples
        sample_indices = np.random.choice(len(self.y_test), 10, replace=False)
        sample_df = pd.DataFrame({
            'Actual': self.y_test.iloc[sample_indices].values,
            'Predicted': self.y_test_pred[sample_indices],
            'Difference': self.y_test.iloc[sample_indices].values - self.y_test_pred[sample_indices]
        })
        sample_df['Difference'] = sample_df['Difference'].round(2)
        sample_df['Actual'] = sample_df['Actual'].round(2)
        sample_df['Predicted'] = sample_df['Predicted'].round(2)
        
        print("\n📋 Sample Predictions (Random Test Samples):")
        print(sample_df.to_string(index=False))
        
        return self.y_train_pred, self.y_test_pred
    
    def evaluate_model(self):
        """Calculate and display evaluation metrics"""
        print("\n" + "="*60)
        print("MODEL EVALUATION METRICS")
        print("="*60)
        
        # Calculate metrics
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        test_r2 = r2_score(self.y_test, self.y_test_pred)
        train_mse = mean_squared_error(self.y_train, self.y_train_pred)
        test_mse = mean_squared_error(self.y_test, self.y_test_pred)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_mae = mean_absolute_error(self.y_train, self.y_train_pred)
        test_mae = mean_absolute_error(self.y_test, self.y_test_pred)
        
        # Display metrics
        print("\n📊 Performance Metrics:")
        print("-" * 60)
        print(f"{'Metric':<20} {'Training':>12} {'Test':>12}")
        print("-" * 60)
        print(f"{'R² Score':<20} {train_r2:>12.4f} {test_r2:>12.4f}")
        print(f"{'MSE':<20} {train_mse:>12.2f} {test_mse:>12.2f}")
        print(f"{'RMSE':<20} {train_rmse:>12.2f} {test_rmse:>12.2f}")
        print(f"{'MAE':<20} {train_mae:>12.2f} {test_mae:>12.2f}")
        print("-" * 60)
        
        print("\n📌 Interpretation:")
        print(f"   • R² Score: The model explains {test_r2*100:.1f}% of the variance in house prices")
        print(f"   • RMSE: On average, predictions are off by ${test_rmse:.2f}K")
        print(f"   • MAE: Average absolute error is ${test_mae:.2f}K")
        
        return {
            'train_r2': train_r2, 'test_r2': test_r2,
            'train_mse': train_mse, 'test_mse': test_mse,
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'train_mae': train_mae, 'test_mae': test_mae
        }
    
    def visualize_results(self):
        """Create comprehensive visualizations"""
        logger.info("Creating visualizations...")
        
        # 1. Actual vs Predicted Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Training set
        axes[0].scatter(self.y_train, self.y_train_pred, alpha=0.5, color='#3498db', 
                       edgecolors='black', linewidth=0.5)
        axes[0].plot([self.y_train.min(), self.y_train.max()], 
                    [self.y_train.min(), self.y_train.max()], 'r--', linewidth=2)
        axes[0].set_xlabel('Actual House Price ($1000s)', fontsize=12)
        axes[0].set_ylabel('Predicted House Price ($1000s)', fontsize=12)
        axes[0].set_title(f'Training Set (R² = {r2_score(self.y_train, self.y_train_pred):.4f})', 
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Test set
        axes[1].scatter(self.y_test, self.y_test_pred, alpha=0.5, color='#e67e22', 
                       edgecolors='black', linewidth=0.5)
        axes[1].plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 'r--', linewidth=2)
        axes[1].set_xlabel('Actual House Price ($1000s)', fontsize=12)
        axes[1].set_ylabel('Predicted House Price ($1000s)', fontsize=12)
        axes[1].set_title(f'Test Set (R² = {r2_score(self.y_test, self.y_test_pred):.4f})', 
                         fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.viz_dir}/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Residuals Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        residuals_train = self.y_train - self.y_train_pred
        residuals_test = self.y_test - self.y_test_pred
        
        # Training residuals
        axes[0].scatter(self.y_train_pred, residuals_train, alpha=0.5, color='#3498db', 
                       edgecolors='black', linewidth=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted House Price ($1000s)', fontsize=12)
        axes[0].set_ylabel('Residuals ($1000s)', fontsize=12)
        axes[0].set_title('Training Set Residuals', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Test residuals
        axes[1].scatter(self.y_test_pred, residuals_test, alpha=0.5, color='#e67e22', 
                       edgecolors='black', linewidth=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted House Price ($1000s)', fontsize=12)
        axes[1].set_ylabel('Residuals ($1000s)', fontsize=12)
        axes[1].set_title('Test Set Residuals', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.viz_dir}/residuals.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Residual Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.histplot(residuals_train, kde=True, color='#3498db', ax=axes[0])
        axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Residuals ($1000s)', fontsize=12)
        axes[0].set_title('Training Set - Residual Distribution', fontsize=14, fontweight='bold')
        
        sns.histplot(residuals_test, kde=True, color='#e67e22', ax=axes[1])
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Residuals ($1000s)', fontsize=12)
        axes[1].set_title('Test Set - Residual Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.viz_dir}/residual_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"✅ All visualizations saved to {self.viz_dir}")
    
    def save_model(self):
        """Save trained model and predictions"""
        logger.info("Saving model and predictions...")
        
        # Save model
        joblib.dump(self.model, f'{self.model_dir}/linear_regression_model.pkl')
        
        # Save coefficients
        coef_df = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Coefficient': self.model.coef_
        }).sort_values('Coefficient', ascending=False)
        coef_df.to_csv(f'{self.model_dir}/coefficients.csv', index=False)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'Actual': self.y_test.values,
            'Predicted': self.y_test_pred,
            'Residual': self.y_test.values - self.y_test_pred
        })
        predictions_df.to_csv(f'{self.model_dir}/test_predictions.csv', index=False)
        
        logger.info(f"✅ Model saved to {self.model_dir}")
    
    def generate_report(self, metrics):
        """Generate comprehensive report"""
        print("\n" + "="*60)
        print("📋 FINAL MODEL REPORT")
        print("="*60)
        
        report = f"""
{"="*60}
LINEAR REGRESSION MODEL - COMPLETE REPORT
{"="*60}

1. DATASET INFORMATION
   • Total samples: {len(self.X_train) + len(self.X_test)}
   • Training samples: {len(self.X_train)} ({len(self.X_train)/(len(self.X_train)+len(self.X_test))*100:.1f}%)
   • Testing samples: {len(self.X_test)} ({len(self.X_test)/(len(self.X_train)+len(self.X_test))*100:.1f}%)
   • Features: {self.X_train.shape[1]}

2. MODEL PERFORMANCE
   {'─'*50}
   {'Metric':<20} {'Training':>12} {'Test':>12}
   {'─'*50}
   {'R² Score':<20} {metrics['train_r2']:>12.4f} {metrics['test_r2']:>12.4f}
   {'MSE':<20} {metrics['train_mse']:>12.2f} {metrics['test_mse']:>12.2f}
   {'RMSE':<20} {metrics['train_rmse']:>12.2f} {metrics['test_rmse']:>12.2f}
   {'MAE':<20} {metrics['train_mae']:>12.2f} {metrics['test_mae']:>12.2f}
   {'─'*50}

3. TOP 5 POSITIVE IMPACT FEATURES
   {'─'*50}
"""
        
        # Top positive coefficients
        coef_df = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Coefficient': self.model.coef_
        }).sort_values('Coefficient', ascending=False)
        
        report += "\n   🔼 Features that INCREASE house price:\n"
        for _, row in coef_df.head(5).iterrows():
            report += f"      • {row['Feature']:15}: +{row['Coefficient']:.4f}\n"
        
        report += "\n   🔽 Features that DECREASE house price:\n"
        for _, row in coef_df.tail(5).iterrows():
            report += f"      • {row['Feature']:15}: {row['Coefficient']:.4f}\n"
        
        report += f"""
   {'─'*50}

4. MODEL EQUATION
   House Price = {self.model.intercept_:.2f}
"""
        for _, row in coef_df.head(3).iterrows():
            report += f"               {row['Coefficient']:+.2f} × {row['Feature']}\n"
        report += f"               + ... (other features)"

        report += f"""
   {'─'*50}

5. INTERPRETATION
   • The model explains {metrics['test_r2']*100:.1f}% of the variance in house prices.
   • Average prediction error (MAE): ${metrics['test_mae']:.2f}K.
   • 95% of predictions are within ±${metrics['test_rmse']*1.96:.2f}K.

6. OUTPUT FILES
   • Model: output/models/linear_regression_model.pkl
   • Coefficients: output/models/coefficients.csv
   • Test predictions: output/models/test_predictions.csv
   • Visualizations: output/visualizations/

{"="*60}
"""
        print(report)
        
        # Save report
        with open(f'{self.model_dir}/model_report.txt', 'w') as f:
            f.write(report)
        
        logger.info(f"✅ Report saved to {self.model_dir}/model_report.txt")
    
    def run_pipeline(self):
        """Run complete training pipeline"""
        print("\n" + "="*60)
        print("🏠 HOUSE PRICE PREDICTION - LINEAR REGRESSION PIPELINE")
        print("="*60)
        
        self.load_data()
        self.explore_target()
        self.train_model()
        coef_df = self.analyze_coefficients()
        self.make_predictions()
        metrics = self.evaluate_model()
        self.visualize_results()
        self.save_model()
        self.generate_report(metrics)
        
        print("\n" + "="*60)
        print("✅✅✅ TASK 2 COMPLETED SUCCESSFULLY! ✅✅✅")
        print("="*60)


if __name__ == "__main__":
    # Run the complete pipeline
    model_pipeline = LinearRegressionModel()
    model_pipeline.run_pipeline()