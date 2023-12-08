# first line: 18
@memory.cache
def update_playground_data(start_date, end_date):
    currency_pairs = ["USDEUR","USDJPY","USDGBP","USDCHF","USDAUD","USDCAD","USDNZD","USDSEK","USDSGD","USDNOK"]
    # Create an empty DataFrame to store the combined data
    combined_currency_data = pd.DataFrame()
    # Fetch data for each currency pair and combine into a single DataFrame
    for pair in currency_pairs:
        currency_data = get_currency_pair_data(pair, start_date, end_date)
        if currency_data is not None:
            combined_currency_data[pair] = currency_data[currency_data.columns[0]]
        time.sleep(0.1)

    #download S&P 500 Data
    # download data
    ticker = '^GSPC'
    spx_data = download_sp500_prices(ticker, start_date, end_date)

    # Define the directory to save the CSV file
    output_directory = '../../data/raw/playground'  # Relative path from the current directory
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
    
    #Process data
    # Define the input file name
    currency_data = 'currency_data.csv'
    # Compute log returns and save the processed data
    compute_log_returns(currency_data,'../../data/raw/playground', '../../data/processed/playground')
    print('Finished processing currency data')
    # Define the input file name
    spx_data = 'spx_data.csv'
    # Compute log returns and save the processed data
    compute_log_returns(spx_data,'../../data/raw/playground', '../../data/processed/playground')