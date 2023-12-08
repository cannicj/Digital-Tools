from download_rawdata_functions import get_currency_pair_data
from download_rawdata_functions import download_sp500_prices
import pandas as pd
import os
from pandas_datareader import DataReader
import datetime
import time

#download all the currency data
print("Downloading data for all needed currencies.")
currency_pairs = ["USDEUR","USDJPY","USDGBP","USDCHF","USDAUD","USDCAD","USDNZD","USDSEK","USDSGD","USDNOK"]
# Create an empty DataFrame to store the combined data
combined_currency_data = pd.DataFrame()
# Fetch data for each currency pair and combine into a single DataFrame
for pair in currency_pairs:
    currency_data = get_currency_pair_data(pair)
    if currency_data is not None:
        combined_currency_data[pair] = currency_data[currency_data.columns[0]]
    time.sleep(1)

#download S&P 500 Data
print("Downloading S&P 500 data.")
# download data
ticker = '^GSPC'
start_date = datetime.datetime(2000, 1, 1)
spx_data = download_sp500_prices()

# Define the directory to save the CSV file
output_directory = '../../data/raw'  # Relative path from the current directory
# Create the directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# save the currency file as .csv
# Define the output file path
output_file = os.path.join(output_directory, 'currency_data.csv')
# Save the combined data to a CSV file in the specified directory
combined_currency_data.to_csv(output_file, index=True, mode = 'w')

# save the spx datafile as .csv
# Define the output file path
output_file = os.path.join(output_directory, 'spx_data.csv')
# Save the combined data to a CSV file in the specified directory
spx_data.to_csv(output_file, index=True, mode = 'w')