# script that takes the downloaded data
# and transforms it into the wanted format: logreturns!

import os
import pandas as pd
import numpy as np

def compute_log_returns(input_file, output_directory):
    # Load the previously saved combined currency data
    combined_currency_data = pd.read_csv(input_file, index_col=0)
    combined_currency_data = combined_currency_data.dropna()
    # Compute log returns for each currency column
    log_returns = pd.DataFrame()
    for column in combined_currency_data.columns:
        log_returns[column] = np.log(combined_currency_data[column]) - np.log(combined_currency_data[column].shift(1))
    log_returns = log_returns.round(5)
    # Define the output file path for processed data
    output_file = os.path.join(output_directory, 'log_returns_currency_data.csv')

    # Save the log returns to a CSV file in the specified directory
    log_returns.to_csv(output_file, index=True, mode = 'w')

# Define the input file path
input_file_path = '../../data/raw/combined_currency_data.csv'

# Define the output directory
output_directory = '../../data/processed'

# Compute log returns and save the processed data
compute_log_returns(input_file_path, output_directory)
