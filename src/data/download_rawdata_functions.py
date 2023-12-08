# download rawdata functions
import pandas as pd
from pandas_datareader import DataReader
import yfinance as yf
import pandas as pd
import datetime

def get_currency_pair_data(currency_pair, start_date = None, end_date = None):
    '''
    function takes the currency pair as input (USD base) and gives the exchange prices as output
    '''
    # Dictionary for currency codes
    currency_codes = {
        "USDEUR": "DEXUSEU",
        "USDJPY": "DEXJPUS",
        "USDGBP": "DEXUSUK",
        "USDCHF": "DEXSZUS",
        "USDAUD": "DEXUSAL",
        "USDCAD": "DEXCAUS",
        "USDHKD": "DEXMAUS",
        "USDNZD": "DEXUSNZ",
        "USDSEK": "DEXSDUS",
        "USDSGD": "DEXSIUS",
        "USDNOK": "DEXNOUS",
        "USDKRW": "DEXKOUS",
        "USDTRY": "DEXMAUS",
        "USDMXN": "DEXMAUS",
        "USDRUB": "DEXRUS"
    }

    # Check if the input is a string and its length is 6
    if not isinstance(currency_pair, str) or len(currency_pair) != 6:
        print("Please provide a string of length 6 for the currency pair, e.g., 'USDCHF'.")
        return None

    if currency_pair[:3] != 'USD':
        currency_pair = currency_pair[3:]+currency_pair[:3]
        confirm = input(
            f"USD will be the base currency. Still continue with {currency_pair}? Press Enter to continue or any other key to abort.")
        if confirm.lower() != '':
            print("Operation aborted.")
            return None

    currency_code = currency_codes.get(currency_pair)
    if currency_code is None:
        print("Currency pair not found.")
        return None

    try:
        if start_date == None or end_date == None :
            start_date = datetime.datetime(2000, 1, 1)
            currency_data = DataReader(currency_code, 'fred', start_date)
        else : 
            currency_data = DataReader(currency_code, 'fred', start_date, end_date)
        #Problem: USD not always base currency. Check if it is, if it is not, take the inverse
        if currency_code[:2] != 'US':
            currency_data = 1/currency_data
        return currency_data
    except Exception as e:
        print("Error:", e)
        return None

# Example usage:
#currency_pair = 'CHFUSD'
#currency_dataframe = get_currency_pair_data(currency_pair)

# SP500 data function from the yahoo finance API
def download_sp500_prices(ticker='^GSPC', start_date=None, end_date = None):
    '''
    Function takes whatever ticker and a start date as input and downloads the respective price data from the yahoo finance API
    ticker default = SP500
    start_date default = 2000-01-01
    '''
    
    try:
        # Download historical data from Yahoo Finance
        if start_date == None or end_date==None:
            data = yf.download(ticker, start="2000-01-01")
        else : 
            data = yf.download(ticker, start=start_date, end = end_date)
        # Extract adjusted closing prices
        sp500_prices = data['Adj Close']
        # create dataframe
        df = pd.DataFrame(sp500_prices)
        df.rename(columns={'Adj Close': 'SP500'}, inplace=True)
        df = df.rename_axis('DATE')
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None


# Example usage:
#ticker = '^GSPC'  # Ticker symbol for S&P 500
#start_date = "2000-01-01"  # Specify the start date
#sp500_df = download_sp500_prices()