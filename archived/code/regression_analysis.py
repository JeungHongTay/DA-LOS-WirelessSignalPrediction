import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Tuple, Optional
from uwb_dataset import import_from_files

def prepare_data(dataset_path='dataset/'):
    """
    Load and prepare data for regression analysis.
    
    Args:
        dataset_path (str): Path to the dataset directory
        
    Returns:
        tuple: Contains:
            - X_train_scaled (np.ndarray): Scaled training features
            - X_test_scaled (np.ndarray): Scaled test features
            - y_train (np.ndarray): Training targets
            - y_test (np.ndarray): Test targets
            - scaler (StandardScaler): Fitted scaler object
    """
    try:
        print("Loading dataset...")
        data = import_from_files(dataset_path)
        
        if data is None or len(data) == 0:
            raise ValueError("Empty dataset received")
            
        print(f"Data shape: {data.shape}")
            
        # First column is measured range (target)
        # Next 14 columns are features (excluding CIR values)
        X = data[:, 1:15]  # Features (excluding measured range and CIR values)
        y = data[:, 0]     # Target (measured range)
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
        
    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        print(f"Make sure the dataset directory exists at: {os.path.abspath(dataset_path)}")
        raise

def analyze_feature_importance(model: RandomForestRegressor, feature_names: list) -> None:
    """
    Analyze and visualize feature importance.
    
    Args:
        model: Trained RandomForestRegressor model
        feature_names: List of feature names
    """
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances for Range Prediction")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                  [feature_names[i] for i in indices], 
                  rotation=45, 
                  ha='right')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        print("\nFeature importance ranking:")
        for f in range(len(importances)):
            print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))
    
    except Exception as e:
        print(f"Error analyzing feature importance: {str(e)}")
        raise

def train_and_evaluate() -> None:
    """Train and evaluate regression model with cross-validation and hyperparameter tuning."""
    try:
        # Prepare data
        X_train, X_test, y_train, y_test, scaler = prepare_data()
        
        # Define feature names
        feature_names = [
            'FP_IDX', 'FP_AMP1', 'FP_AMP2', 'FP_AMP3', 'STDEV_NOISE',
            'CIR_PWR', 'MAX_NOISE', 'RXPACC', 'CH', 'FRAME_LEN',
            'PREAM_LEN', 'BITRATE', 'PRFR', 'LOS_FLAG'
        ]
        
        # Perform hyperparameter tuning
        print("\nPerforming hyperparameter tuning...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        base_model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(base_model, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Train model with best parameters
        best_model = grid_search.best_estimator_
        
        # Perform cross-validation
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
        print("\nCross-validation scores:")
        print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print("\nModel Performance:")
        print(f"Root Mean Square Error: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Analyze feature importance
        analyze_feature_importance(best_model, feature_names)
        
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', 
                lw=2)
        plt.xlabel('Actual Range')
        plt.ylabel('Predicted Range')
        plt.title('Actual vs Predicted Range')
        plt.tight_layout()
        plt.savefig('actual_vs_predicted.png')
        plt.close()
        
        # Plot residuals
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.xlabel('Predicted Range')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig('residuals.png')
        plt.close()
        
        # Save the model and scaler
        joblib.dump(best_model, 'rf_model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        
    except Exception as e:
        print(f"Error in train_and_evaluate: {str(e)}")
        raise

if __name__ == '__main__':
    train_and_evaluate()