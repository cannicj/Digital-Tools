import pandas as pd
import sys 
import os
import random
import matplotlib.pyplot as plt
import datetime as datetime
sys.path.append('../../src/models')
sys.path.append('../../src/visualization')
sys.path.append('../../src/data')
from support_vector_machine import support_vector_machine
from randomforest_classifier import randomforest_classifier
from decision_tree_classifier import decision_tree_classifier
from update_playground_data import update_playground_data
from plot_results import plot_results
from combine_tables import combine_tables


def update_plot_playground(currencies, include_sp500, lag, train_size, random_seed, dtc_active, rfc_active, svm_active, dtc_long_only, rfc_long_only, svm_long_only,
                           dtc_max_depth, rfc_max_depth, rfc_trees, rfc_leaves, fig, plot_output_widget, start_date, end_date):
    update_playground_data(start_date, end_date)
    #Import our data
    log_returns_currencies = pd.read_csv("../../data/processed/playground/log_returns_currency_data.csv")
    log_returns_spx = pd.read_csv("../../data/processed/playground/log_returns_spx_data.csv")
    dataframe = pd.merge(log_returns_currencies, log_returns_spx.iloc[1:], on='DATE', how='inner')

    #If we have random seed set to true, generate a random seed, else set seed to 42 (our standard value)
    if random_seed == True:
        seed = random.randint(0, 4294967295)
    else :
        seed = 42
        
    # Train the models that are set to active
    active_models = []
    if dtc_active == True:
        dt_accuracies, dt_results = decision_tree_classifier(dataframe, currencies, include_sp500, lag, train_size, seed, dtc_long_only, dtc_max_depth)
        active_models.append(dt_results)
    if rfc_active == True:
        rf_accuracies, rf_results = randomforest_classifier(dataframe, currencies, include_sp500, lag, train_size, seed, rfc_long_only, rfc_trees, rfc_max_depth, rfc_leaves)
        active_models.append(rf_results)
    if svm_active == True:
        svm_accuracies, svm_results = support_vector_machine(dataframe, currencies, include_sp500, lag, train_size, seed, svm_long_only)
        active_models.append(svm_results)
    #Combine the results tables of the active trained models and plot the final results
    results = combine_tables(active_models)
    plot_results(results, include_sp500, fig, plot_output_widget)
    current_time = datetime.datetime.now()
    print(f"Finished updating plot at: {current_time}")
    

