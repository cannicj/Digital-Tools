# download rawdata functions
import pandas as pd
from pandas_datareader import DataReader
import datetime

def get_currency_pair_data(currency_pair):
    # function takes the currency pair as input (USD base) and gives the data as output
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
        start_date = datetime.datetime(2000, 1, 1)
        currency_data = DataReader(currency_code, 'fred', start_date)
        return currency_data
    except Exception as e:
        print("Error:", e)
        return None

# Example usage:
#currency_pair = 'CHFUSD'
#currency_dataframe = get_currency_pair_data(currency_pair)



