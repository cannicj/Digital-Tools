# script that takes the downloaded data
# and transforms it into the wanted format: logreturns!

import os
import pandas as pd
import numpy as np

def compute_log_returns(input_file, input_directory_path = None, output_directory_path = None):
    # Load the previously saved combined currency data
    if input_directory_path == None:
        data = pd.read_csv(f'../../data/raw/{input_file}', index_col=0)
    else : 
        data = pd.read_csv(f'{input_directory_path}/{input_file}', index_col=0)
    data = data.dropna()
    # Compute log returns for each currency column
    log_returns = pd.DataFrame()
    for column in data.columns:
        log_returns[column] = np.log(data[column]) - np.log(data[column].shift(1))
    log_returns = log_returns.round(5)
    # Define the output file path for processed data
    if output_directory_path == None : 
        output_file = os.path.join('../../data/processed', f'log_returns_{input_file}')
    else :
        output_file = os.path.join(f'{output_directory_path}', f'log_returns_{input_file}')

    # Save the log returns to a CSV file in the specified directory
    log_returns.to_csv(output_file, index=True, mode = 'w')

print("Processing currency data.")
# Define the input file name
currency_data = 'currency_data.csv'
# Compute log returns and save the processed data
compute_log_returns(currency_data)

print("Processing SPX data.")
# Define the input file name
spx_data = 'spx_data.csv'
# Compute log returns and save the processed data
compute_log_returns(spx_data)


