from download_rawdata_functions import get_currency_pair_data
import pandas as pd
import os

currency_pairs = ["USDEUR","USDJPY","USDGBP","USDCHF","USDAUD","USDCAD","USDHKD","USDNZD","USDSEK","USDSGD","USDNOK","USDKRW","USDTRY","USDMXN","USDRUB"]
# Create an empty DataFrame to store the combined data
combined_currency_data = pd.DataFrame()

# Fetch data for each currency pair and combine into a single DataFrame
for pair in currency_pairs:
    currency_data = get_currency_pair_data(pair)
    if currency_data is not None:
        combined_currency_data[pair] = currency_data[currency_data.columns[0]]

# Define the directory to save the CSV file
output_directory = '../../data/raw'  # Relative path from the current directory

# Create the directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Define the output file path
output_file = os.path.join(output_directory, 'combined_currency_data.csv')

# Save the combined data to a CSV file in the specified directory
combined_currency_data.to_csv(output_file, index=True)