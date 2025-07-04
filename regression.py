"""
Boston Housing Price Prediction - Main Implementation
This script implements and compares multiple regression models for predicting house prices
"""

import os
import sys
from utils import (
    load_data, explore_data, preprocess_data, get_regression_models,
    compare_models, get_hyperparameter_grids, hyperparameter_tuning,
    save_results, plot_results, print_summary
)


def main():
    """
    Main function to run the complete machine learning workflow
    """
    print("BOSTON HOUSING PRICE PREDICTION")

    
    # Step 1: Load the data
    print("\n1. Loading Boston Housing Dataset...")
    X, y, feature_names = load_data()
    
    # Step 2: Explore the data
    print("\n2. Exploring the dataset...")
    explore_data(X, y)
    
    # Step 3: Preprocess the data
    print("\n3. Preprocessing the data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y, scale_features=True)
    
    # Step 4: Get regression models
    print("\n4. Initializing regression models...")
    models = get_regression_models()
    print(f"Models to compare: {list(models.keys())}")
    
    # Step 5: Compare basic models (for reg_branch)
    print("\n5. Comparing basic regression models...")
    comparison_df_basic, results_basic = compare_models(models, X_train, X_test, y_train, y_test)
    
    # Step 6: Perform hyperparameter tuning
    
    print("\n6. Performing hyperparameter tuning...")
        
    # Get parameter grids
    param_grids = get_hyperparameter_grids()
        
    # Perform hyperparameter tuning
    best_models = hyperparameter_tuning(models, param_grids, X_train, y_train)
        
    # Compare tuned models
    print("\n7. Comparing hyperparameter-tuned models...")
    comparison_df_tuned, results_tuned = compare_models(best_models, X_train, X_test, y_train, y_test)
        
    # Save tuned results
    save_results(comparison_df_tuned, 'hypertuned_model_results.csv')
        
    # Print comparison between basic and tuned models
    print("BASIC vs HYPERPARAMETER-TUNED COMPARISON")
    
        
    for model_name in models.keys():
        basic_result = comparison_df_basic[comparison_df_basic['Model'] == model_name]
        tuned_result = comparison_df_tuned[comparison_df_tuned['Model'] == model_name]
            
        if not basic_result.empty and not tuned_result.empty:
            basic_mse = basic_result['Test_MSE'].iloc[0]
            tuned_mse = tuned_result['Test_MSE'].iloc[0]
            basic_r2 = basic_result['Test_R2'].iloc[0]
            tuned_r2 = tuned_result['Test_R2'].iloc[0]
                
            mse_improvement = ((basic_mse - tuned_mse) / basic_mse) * 100
            r2_improvement = ((tuned_r2 - basic_r2) / basic_r2) * 100
                
            print(f"\n{model_name}:")
            print(f"  Basic    - MSE: {basic_mse:.4f}, R²: {basic_r2:.4f}")
            print(f"  Tuned    - MSE: {tuned_mse:.4f}, R²: {tuned_r2:.4f}")
            print(f"  Improvement - MSE: {mse_improvement:+.2f}%, R²: {r2_improvement:+.2f}%")
        
        # Use tuned results for final summary
    final_comparison_df = comparison_df_tuned

    
    # Step 7: Save results
    print("\n" + ("7") + ". Saving results...")
    save_results(comparison_df_basic, 'basic_model_results.csv')
    
    # Step 8: Create visualizations
    print("\n" + ("8") + ". Creating visualizations...")
    try:
        plot_results(final_comparison_df)
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")
        print("This might be due to display issues in CI environment")
    
    # Step 9: Print summary
    print_summary(final_comparison_df)
    

    print("ANALYSIS COMPLETED!")
    
    return final_comparison_df

if __name__ == "__main__":
        main()