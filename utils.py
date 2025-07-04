"""
Utility functions for Boston Housing Price Prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """
    Load Boston Housing dataset
    
    Returns:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        feature_names (list): List of feature names
    """
    try:

        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        # now we split this into data and target
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT'
        ]


        # Create a DataFrame
        X = pd.DataFrame(data, columns=feature_names)
        y = pd.Series(target, name='MEDV')
        #df[’MEDV’] = target # here MEDV is our target variable
        print("Boston Housing dataset loaded successfully")
        print(f"Dataset shape: {X.shape}")
        print(f"Target variable: {y.name}")
        return X,y,feature_names
    
    except ImportError:
        print("Error: Could not load Boston Housing dataset")
        print("Note: Boston Housing dataset is deprecated in newer scikit-learn versions")
        return None, None, None


def explore_data(X, y):
    """
    Perform exploratory data analysis

    """
    print("EXPLORATORY DATA ANALYSIS")
    
    # Basic info
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    
    # Missing values
    print(f"\nMissing values in features: {X.isnull().sum().sum()}")
    print(f"Missing values in target: {y.isnull().sum()}")
    
    # Statistical summary
    print("\nTarget variable statistics:")
    print(f"Mean: {y.mean():.2f}")
    print(f"Std: {y.std():.2f}")
    print(f"Min: {y.min():.2f}")
    print(f"Max: {y.max():.2f}")
    
    # Feature correlation with target
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    print(f"\nTop 5 features correlated with target:")
    for i, (feature, corr) in enumerate(correlations.head().items()):
        print(f"{i+1}. {feature}: {corr:.3f}")


def preprocess_data(X, y, test_size=0.2, random_state=42, scale_features=True):
    """
    Preprocess the data: split and optionally scale
    
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = None
    if scale_features:
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Convert back to DataFrame for consistency
        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)
    
    print(f"\nData preprocessed successfully")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Feature scaling: {'Applied' if scale_features else 'Not applied'}")
    
    return X_train, X_test, y_train, y_test, scaler


def get_regression_models():
 
    models = {
        'Ridge': Ridge(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Lasso': Lasso(),
        'SVR': SVR(),
    }
    return models


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_mse = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    results = {
        'model_name': model_name,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_mse': cv_mse,
        'cv_std': cv_std,
        'model': model
    }
    
    return results


def compare_models(models, X_train, X_test, y_train, y_test):
    """
    Compare multiple regression models
    """
    results = []
    
    print("MODEL EVALUATION RESULTS")

    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        result = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(result)
        
        # Print results
        print(f"{name} Results:")
        print(f"  Train MSE: {result['train_mse']:.4f}")
        print(f"  Test MSE:  {result['test_mse']:.4f}")
        print(f"  Train R²:  {result['train_r2']:.4f}")
        print(f"  Test R²:   {result['test_r2']:.4f}")
        print(f"  CV MSE:    {result['cv_mse']:.4f} (±{result['cv_std']:.4f})")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Train_MSE': r['train_mse'],
            'Test_MSE': r['test_mse'],
            'Train_R2': r['train_r2'],
            'Test_R2': r['test_r2'],
            'CV_MSE': r['cv_mse']
        }
        for r in results
    ])
    
    return comparison_df, results


def get_hyperparameter_grids():
    """
    Get hyperparameter grids for tuning
    
    """
    param_grids = {
        'Ridge': {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'fit_intercept': [True, False],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr']
        },
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1,2],
            'max_features': ['sqrt']
        },
        'Lasso': {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
            'fit_intercept': [True, False],
            'selection': ['cyclic', 'random'],
            'max_iter': [1000, 5000, 10000]
        },
       'SVR': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'epsilon': [0.01, 0.1, 0.2, 0.5],
            'kernel': ['rbf', 'linear', 'poly']
        },
    }
    return param_grids


def hyperparameter_tuning(models, param_grids, X_train, y_train, cv=5):
    """
    Perform hyperparameter tuning using GridSearchCV
    """
    best_models = {}
    
    print("HYPERPARAMETER TUNING")

    for name, model in models.items():
        print(f"\nTuning {name}...")
        
        # Get parameter grid
        param_grid = param_grids.get(name, {})
        
        #if not param_grid:
        #    print(f"No parameter grid found for {name}, using default parameters")
        #    best_models[name] = model
        #    continue
        
        if name == 'SVR':
            from sklearn.model_selection import RandomizedSearchCV
            grid_search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=30,  # Limit SVR to 30 random combinations
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=42
            )
        else:
        # Perform grid search
            grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store best model
        best_models[name] = grid_search.best_estimator_
        
        # Print results
        print(f"Best parameters for {name}:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"Best CV MSE: {-grid_search.best_score_:.4f}")
    
    return best_models


def save_results(comparison_df, filename='model_comparison_results.csv'):
    """
    Save comparison results to CSV file
    """
    comparison_df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")


def plot_results(comparison_df, save_plot=True):
    """
    Create visualization of model comparison results
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # MSE comparison
    axes[0].bar(comparison_df['Model'], comparison_df['Test_MSE'], 
                color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0].set_title('Test MSE Comparison')
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].tick_params(axis='x', rotation=45)
    
    # R² comparison
    axes[1].bar(comparison_df['Model'], comparison_df['Test_R2'], 
                color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[1].set_title('Test R² Score Comparison')
    axes[1].set_ylabel('R² Score')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('modelcomp.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'modelcomp.png'")
    
    plt.show()


def print_summary(comparison_df):
    """
    Print a summary of the best performing model
    """
    print("SUMMARY")
    
    # Best model by Test MSE (lower is better)
    best_mse_idx = comparison_df['Test_MSE'].idxmin()
    best_mse_model = comparison_df.loc[best_mse_idx]
    
    # Best model by Test R² (higher is better)
    best_r2_idx = comparison_df['Test_R2'].idxmax()
    best_r2_model = comparison_df.loc[best_r2_idx]
    
    print(f"Best model by MSE: {best_mse_model['Model']}")
    print(f"  Test MSE: {best_mse_model['Test_MSE']:.4f}")
    print(f"  Test R²:  {best_mse_model['Test_R2']:.4f}")
    
    print(f"\nBest model by R²: {best_r2_model['Model']}")
    print(f"  Test MSE: {best_r2_model['Test_MSE']:.4f}")
    print(f"  Test R²:  {best_r2_model['Test_R2']:.4f}")
    
    if best_mse_model['Model'] == best_r2_model['Model']:
        print(f"\nOverall best model: {best_mse_model['Model']}")
    else:
        print(f"\nBest models differ by metric - consider domain requirements")