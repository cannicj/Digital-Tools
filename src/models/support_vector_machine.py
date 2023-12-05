#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def support_vector_machine(dataframe, currencies=None, include_sp500=True,lag=1,train_size=0.75, seed=42, long_only=False):
    """
    Trains a Support Vector Classifier (SVC) on financial data and evaluates its performance.

    This function processes financial data, potentially including S&P 500 information, and applies a Support Vector Classifier to predict binary outcomes. It also calculates the model's accuracy and plots the cumulative returns for comparison.

    Parameters:
    - dataframe (pd.DataFrame): The dataset containing the financial data. It should have a 'DATE' column, currency data columns, and optionally an 'SP500' column.
    - currencies (list of str, optional): A list of currency columns to include in the analysis. If None, all available currencies in the dataframe will be used.
    - include_sp500 (bool): Determines whether to include the S&P 500 data in the analysis. Defaults to True.
    - lag (int): The number of periods by which to lag the response variable for prediction. Defaults to 1.
    - train_size (float): The proportion of the dataset to use for training the model. The rest will be used for testing. Defaults to 0.75.
    - seed: The seed can be set manually such that the results are reproducible. Default is 42.

    Returns:
    - tuple: A tuple containing a DataFrame of model accuracies and another DataFrame with the cumulative returns for both the 'Long' strategy and the SVC model.
    """
    # Selecting columns based on currencies if provided
    if currencies is not None:
        unavailable_currencies = [currency for currency in currencies if currency not in dataframe.columns]
        if unavailable_currencies:
            available_currencies = ', '.join([col for col in dataframe.columns if col not in ['DATE', 'SP500']])
            unavailable_currencies_str = ', '.join(unavailable_currencies)
            print(f"Sorry, {unavailable_currencies_str} is not an available currency pair. Please choose currency pairs from: {available_currencies}")
            return None

    # Setting up response and regressor variables
    y1 = dataframe.iloc[lag:, -1]  # Assuming the last column is the response variable
    y_dates = dataframe.iloc[lag:, 0]
    X1 = dataframe.iloc[:-lag, 1:-1]  # Excluding Date and response variable

    # Including or excluding SP500 based on the flag
    if include_sp500:
        x1_spx = dataframe.iloc[:-lag, -1]
        X1 = pd.concat([X1, x1_spx], axis=1)

    # Calculating binary response
    y1_exp = np.exp(y1) - 1
    y1_binary = (y1_exp > 0).astype(int)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X1, y1_binary, random_state=seed, shuffle=False, test_size=1-train_size)

    # Setting up dates
    y_test_dates = y_dates[y_test.index[0]-lag:]
    
    # Training the SVC model
    svc_clf = SVC(kernel="rbf", random_state=seed)
    svc_clf.fit(X_train, y_train)
    y_pred_train_svc = svc_clf.predict(X_train)
    y_pred_test_svc = svc_clf.predict(X_test)
    
    # Changing all 0 to -1 for return calculation if long_only is set to false
    if long_only==False:
        y_pred_test_svc[np.where(y_pred_test_svc == 0)] = -1

    # Calculating returns
    y1_ret = y1_exp
    y_bench = y1_ret[y_test.index[0] - lag:]
    y_long = np.cumsum(y_bench)
    y_svc = np.cumsum(y_bench * y_pred_test_svc)

    # Calculating accuracies
    accuracies = pd.DataFrame({
        "Classifiers": ["SVC"],
        "in sample": [accuracy_score(y_train, y_pred_train_svc)],
        "out of sample": [accuracy_score(y_test, y_pred_test_svc)]
    }).set_index('Classifiers')

    test_perform = pd.DataFrame({"SP500": y_long, "SVC": y_svc})

    cumreturns = pd.concat([y_test_dates, test_perform], axis=1)

    return accuracies, cumreturns


# In[ ]:




