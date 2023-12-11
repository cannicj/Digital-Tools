import pandas as pd
import numpy as np
import datetime
from datetime import date

def run_simulations(dataframe, currencies, include_sp500, lag, train_size, long_only, num_simulations):
    # Initialize lists to store results
    svm_results_list = pd.DataFrame()
    rf_results_list = pd.DataFrame()
    dt_results_list = pd.DataFrame()

    # Run simulations for each model
    for i in range(num_simulations):
        seed = np.random.randint(0, 10000)  # Random seed for each simulation

        # SVM model
        _, svm_results = support_vector_machine(dataframe, currencies, include_sp500, lag, train_size, seed, long_only)
        if i == 0:
            svm_dates = svm_results['DATE']
            svm_results_list = pd.concat([svm_results_list, svm_dates], axis=1)
        svm_relevant_column = svm_results.iloc[:, -1].rename(f'SVM Simulation {i+1}')
        svm_results_list = pd.concat([svm_results_list, svm_relevant_column], axis=1)

        # Random Forest model
        _, rf_results = randomforest_classifier(dataframe, currencies, include_sp500, lag, train_size, seed, long_only)
        if i == 0:
            rf_dates = rf_results['DATE']
            rf_results_list = pd.concat([rf_results_list, rf_dates], axis=1)
        rf_relevant_column = rf_results.iloc[:, -1].rename(f'RF Simulation {i+1}')
        rf_results_list = pd.concat([rf_results_list, rf_relevant_column], axis=1)

        # Decision Tree model
        _, dt_results = decision_tree_classifier(dataframe, currencies, include_sp500, lag, train_size, seed, long_only, max_depth=100)
        if i == 0:
            dt_dates = dt_results['DATE']
            dt_results_list = pd.concat([dt_results_list, dt_dates], axis=1)
        dt_relevant_column = dt_results.iloc[:, -1].rename(f'DT Simulation {i+1}')
        dt_results_list = pd.concat([dt_results_list, dt_relevant_column], axis=1)
    
    svm_results_list['DATE'] = pd.to_datetime(svm_results_list['DATE'], format='%Y-%m-%d')
    rf_results_list['DATE'] = pd.to_datetime(rf_results_list['DATE'], format='%Y-%m-%d')
    dt_results_list['DATE'] = pd.to_datetime(dt_results_list['DATE'], format='%Y-%m-%d')
    
    return svm_results_list, rf_results_list, dt_results_list

