import os
import sys
from utils import (
    load_data, explore_data, preprocess_data, get_regression_models,
    compare_models, get_hyperparameter_grids, hyperparameter_tuning,
    save_results, plot_results, print_summary
)


def main():
  
    print("BOSTON HOUSING PRICE PREDICTION")
   
    # Step 1: Loading the data
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

    # Step 6: Save results
    print("\n" + ("6") + ". Saving results...")
    save_results(comparison_df_basic, 'basic_model_results.csv')
    
    # Step 7: Create visualizations
    print("\n" + ("7") + ". Creating visualizations...")
    plot_results(comparison_df_basic)
    
    # Step 8: Print summary
    print_summary(comparison_df_basic)
    
    
    print("ANALYSIS COMPLETED!")
    
    return comparison_df_basic

if __name__ == "__main__":
        main()