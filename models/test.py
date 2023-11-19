from randomforest_classifier import randomforest_classifier
import pandas as pd

# Example usage:
# Assume df is your dataframe
#Import our data
log_returns_currencies = pd.read_csv("../data/processed/log_returns_currency_data.csv")
log_returns_spx = pd.read_csv("../data/processed/log_returns_spx_data.csv")
dataframe = pd.merge(log_returns_currencies, log_returns_spx.iloc[1:], on='DATE', how='inner')
print(dataframe.columns)
results = randomforest_classifier(dataframe, currencies=['USDCHF', 'USDJPY', 'USDEUR', 'USDGBP'], train_size=0.75)
#print(results)